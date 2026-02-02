
    void controlLoop()
    {
        // 检查是否收到必要数据
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                "等待里程计数据...");
            return;
        }
        if (!trajectory_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                "等待参考轨迹数据...");
            return;
        }

       // 线程安全获取数据
        nav_msgs::msg::Odometry current_odom;
        autoware_auto_planning_msgs::msg::Trajectory current_traj;
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            current_odom = current_odometry_;
            // 里程计数据新鲜度检查
            auto time_diff = this->now() - current_odom.header.stamp;
            if (time_diff.seconds() > 1.0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                    "里程计数据过期: %.2f秒", time_diff.seconds());
                return;
            }
        }

        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            current_traj = current_trajectory_;
        }

        // 轨迹数据新鲜度检查（移到锁外面）
        if (!current_traj.points.empty()) {
            auto time_diff = this->now() - current_traj.header.stamp;
            if (time_diff.seconds() > 1.0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                    "轨迹数据过期: %.2f秒", time_diff.seconds());
                return;
            }
        }
            
       // 添加轨迹点有效性检查
        if (current_traj.points.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "接收到空的轨迹点");
            return;
        }

        // 轨迹格式转换 - 将Autoware轨迹格式转换为内部使用的格式
        std::vector<autoware::mpc_lateral_controller::MPCTracker::StateVector> internal_traj;
        if (!mpc_tracker_->convertAutowareTrajectoryToInternal(current_traj, internal_traj)) {
            RCLCPP_ERROR(this->get_logger(), "参考轨迹格式转换失败！");
            return;
        }
        
        // 准备MPC输入/输出变量
        autoware_auto_control_msgs::msg::AckermannLateralCommand dummy_steer;
        autoware_auto_control_msgs::msg::AckermannLateralCommand ackermann_output;
        autoware_auto_planning_msgs::msg::Trajectory predicted_trajectory;
        autoware_adapi_v1_msgs::msg::Float32MultiArrayStamped debug_values;
        debug_values.header.frame_id = "base_link";


        // 设置参考轨迹 - 将转换后的轨迹传递给MPC控制器
        mpc_tracker_->setReferenceTrajectory(internal_traj);

        // 执行MPC计算 - 基于当前状态和参考轨迹计算最优控制命令
        bool mpc_success = mpc_tracker_->calculateMPC(
            dummy_steer,
            current_odom,
            ackermann_output,
            predicted_trajectory,
            debug_values
        );

        // MPC计算成功：发布控制指令
        if (mpc_success) {
            // 发布备用阿克曼指令 - 用于兼容传统车辆模型
            ackermann_output.stamp = this->now();
            control_cmd_pub_->publish(ackermann_output);

        // 发布全向速度指令 - 麦克纳姆轮特有的速度控制指令
        geometry_msgs::msg::Twist mecanum_twist;
        if (debug_values.data.size() >= 3) {
            mecanum_twist.linear.x = debug_values.data[0];  // X方向线速度
            mecanum_twist.linear.y = debug_values.data[1];  // Y方向线速度
            mecanum_twist.angular.z = debug_values.data[2]; // 角速度
            mecanum_twist.header.stamp = this->now();
            mecanum_twist.header.frame_id = "base_link";
            twist_cmd_pub_->publish(mecanum_twist);
            RCLCPP_DEBUG(this->get_logger(), 
                        "全向控制指令：vx=%.2f m/s, vy=%.2f m/s, omega=%.2f rad/s",
                        mecanum_twist.linear.x, mecanum_twist.linear.y, mecanum_twist.angular.z);
        } else {
            RCLCPP_WARN(this->get_logger(), "调试数据不完整！无法发布全向速度指令");
             }

            // 发布轮速指令 - 根据全向运动指令计算各轮子的速度
            if (wheel_speeds_pub_) {
                autoware::mpc_lateral_controller::MPCTracker::ControlVector control_vec(
                    debug_values.data[0], debug_values.data[1], debug_values.data[2]);
                auto wheel_speeds = mpc_tracker_->computeWheelSpeeds(control_vec);
                if (wheel_speeds.size() == 4) {
                    std_msgs::msg::Float32MultiArray wheel_speeds_msg;
                    wheel_speeds_msg.data = {
                        static_cast<float>(wheel_speeds[0]),  // 前左轮速度
                        static_cast<float>(wheel_speeds[1]),  // 前右轮速度
                        static_cast<float>(wheel_speeds[2]),  // 后左轮速度
                        static_cast<float>(wheel_speeds[3])   // 后右轮速度
                    };
                    wheel_speeds_pub_->publish(wheel_speeds_msg);
                    RCLCPP_DEBUG(this->get_logger(), 
                                "轮速指令（rad/s）：FL=%.2f, FR=%.2f, RL=%.2f, RR=%.2f",
                                wheel_speeds[0], wheel_speeds[1], wheel_speeds[2], wheel_speeds[3]);
                } else {
                    RCLCPP_WARN(this->get_logger(), "轮速计算结果无效！需返回4个轮速值");
                }
            }

            // 发布预测轨迹 - 显示MPC预测的未来路径，用于可视化和调试
            predicted_trajectory.header.stamp = this->now();
            predicted_trajectory.header.frame_id = "map";
            predicted_trajectory_pub_->publish(predicted_trajectory);

            // 发布调试数据 - 包含控制过程中的关键参数，便于分析和调试
            debug_values.header.stamp = this->now();
            debug_values_pub_->publish(debug_values);

        } else {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                "MPC计算失败！");  // MPC优化求解失败
        }
    }
