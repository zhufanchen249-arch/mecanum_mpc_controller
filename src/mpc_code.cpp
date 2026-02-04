#include <memory>
#include <mutex>
#include <string>
#include <Eigen/Dense>
// ROS2核心头文件
#include "rclcpp/rclcpp.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
// 算法层头文件
#include "mpc.hpp"
// 消息类型头文件
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_planning_msgs/msg/trajectory.hpp"
// 【修正】使用新版 Control 消息
#include "autoware_control_msgs/msg/control.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std::chrono_literals;
using autoware::mpc_lateral_controller::MPCTracker;
// 【修正】统一使用 Control
using ControlCmd = autoware_control_msgs::msg::Control;
using Trajectory = autoware_planning_msgs::msg::Trajectory;
using Odometry = nav_msgs::msg::Odometry;
using Twist = geometry_msgs::msg::Twist;
using Float32MultiArray = std_msgs::msg::Float32MultiArray;

using std::placeholders::_1;

class MPCControllerNode : public rclcpp::Node {
public:
    MPCControllerNode() : Node("mpc_controller_mecanum") {
        // 1. 声明节点参数
        this->declare_parameter("config_file", "mecanum_car.yaml");
        this->declare_parameter("wheelbase", 0.3);
        this->declare_parameter("control_rate", 20.0);
        this->declare_parameter("enable_ackermann_deception", true); // 新增：启用阿克曼欺骗模式
        this->declare_parameter("virtual_wheelbase", 0.3); // 虚拟轴距用于阿克曼计算
        
        // 2. 获取参数
        std::string config_file = this->get_parameter("config_file").as_string();
        double wheelbase = this->get_parameter("wheelbase").as_double();
        double control_rate = this->get_parameter("control_rate").as_double();
        enable_deception_ = this->get_parameter("enable_ackermann_deception").as_bool();
        virtual_wheelbase_ = this->get_parameter("virtual_wheelbase").as_double();

        if (config_file.find("/") == std::string::npos) {
            std::string pkg_share = ament_index_cpp::get_package_share_directory("mecanum_mpc_controller");
            config_file = pkg_share + "/param/" + config_file;
            RCLCPP_INFO(this->get_logger(), "配置文件路径：%s", config_file.c_str());
        }

        // 3. 初始化MPC算法核心
        try {
            mpc_tracker_ = std::make_unique<MPCTracker>(config_file, wheelbase);
            RCLCPP_INFO(this->get_logger(), "MPC跟踪器初始化成功！");
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "MPC跟踪器初始化失败：%s", e.what());
            rclcpp::shutdown();
            return;
        }

        // 4. 创建订阅器
        sub_odom_ = this->create_subscription<Odometry>(
            "input/odometry", 10, std::bind(&MPCControllerNode::onOdom, this, _1));
        sub_traj_ = this->create_subscription<Trajectory>(
            "input/trajectory", 10, std::bind(&MPCControllerNode::onTraj, this, _1));

        // 5. 创建发布者
        // 【修正】这里直接发布 Control（用于欺骗Autoware）
        pub_control_cmd_ = this->create_publisher<ControlCmd>("output/control_cmd", 10);
        // 真正的麦克纳姆轮控制指令
        pub_mecanum_twist_ = this->create_publisher<Twist>("output/mecanum_twist", 10);        
        pub_pred_traj_ = this->create_publisher<Trajectory>("output/predicted_trajectory", 10);
        pub_debug_ = this->create_publisher<Float32MultiArray>("output/mpc_debug", 10);
        pub_wheel_speeds_ = this->create_publisher<Float32MultiArray>("output/mecanum_wheel_speeds", 10);

        // 6. 创建定时器
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / control_rate),
            std::bind(&MPCControllerNode::mpcControlLoop, this)
        );

        RCLCPP_INFO(this->get_logger(), "麦克纳姆轮MPC控制器节点启动成功！控制频率：%.1fHz", control_rate);
        RCLCPP_INFO(this->get_logger(), "阿克曼欺骗模式：%s", enable_deception_ ? "启用" : "禁用");
    }

private:
    // 控制模式标志
    bool enable_deception_;
    double virtual_wheelbase_;
    
    // 添加一个成员变量保存上一次控制指令
    ControlCmd last_control_;
    
    void onOdom(const Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        current_odom_ = *msg;
        has_odom_ = true;
    }

    void onTraj(const Trajectory::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        current_traj_ = *msg;
        has_traj_ = true;
    }

    void mpcControlLoop() {
        if (!has_odom_ || !has_traj_) {
            return; // 暂不打印等待日志，避免刷屏
        }

        Odometry odom;
        Trajectory traj;
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            odom = current_odom_;
            traj = current_traj_;
        }

        // 1. 轨迹转换
        std::vector<Eigen::Matrix<double, 6, 1>> inner_traj;
        if (!mpc_tracker_->convertAutowareTrajectoryToInternal(traj, inner_traj) || inner_traj.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "轨迹转换失败");
            return;
        }

        // 2. 设置参考轨迹
        if (!mpc_tracker_->setReferenceTrajectory(inner_traj)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "设置参考轨迹失败");
            return;
        }
        
        // 3. 构造全0参考控制
        std::vector<MPCTracker::ControlVector> ref_control(inner_traj.size(), MPCTracker::ControlVector::Zero());
        if (!mpc_tracker_->setReferenceControl(ref_control)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "设置参考控制失败");
            return;
        }

        // 4. 准备变量
        ControlCmd mpc_output_cmd;
        Trajectory pred_traj_msg;
        Float32MultiArray debug_msg;

        // 5. 执行计算 - 使用上一次的控制指令
        if (!mpc_tracker_->calculateMPC(last_control_, odom, mpc_output_cmd, pred_traj_msg, debug_msg)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "MPC求解失败");
            return;
        }

        // 6. 保存当前控制指令
        last_control_ = mpc_output_cmd;

        // 7. 【核心修改】生成欺骗Autoware的阿克曼控制指令
        if (enable_deception_) {
            ControlCmd ackermann_cmd = generateAckermannCommand(mpc_output_cmd, debug_msg);
            ackermann_cmd.stamp = this->now();
            ackermann_cmd.control_time = this->now();
            pub_control_cmd_->publish(ackermann_cmd);
            RCLCPP_DEBUG(this->get_logger(), "发布阿克曼欺骗指令: velocity=%.3f, steering=%.3f", 
                        ackermann_cmd.longitudinal.velocity, ackermann_cmd.lateral.steering_tire_angle);
        }

        // 8. 发布真正的麦克纳姆轮控制指令
        publishMecanumControl(debug_msg);

        // 9. 发布辅助话题
        publishAuxTopics(mpc_output_cmd, pred_traj_msg, debug_msg);
    }

    /**
     * @brief 生成欺骗Autoware的阿克曼控制指令
     * 将麦克纳姆轮的[vx, vy, omega]转换为Autoware期望的阿克曼模型指令
     */
    ControlCmd generateAckermannCommand(const ControlCmd& mpc_cmd, const Float32MultiArray& debug_data) {
        ControlCmd ackermann_cmd;
        
        // 从调试数据中提取真实的麦克纳姆轮控制输出
        // debug_data格式: [x, y, theta, vx_actual, vy_actual, omega_actual]
        if (debug_data.data.size() >= 6) {
            double vx_actual = debug_data.data[3];  // 真实的纵向速度
            double vy_actual = debug_data.data[4];  // 真实的横向速度
            double omega_actual = debug_data.data[5]; // 真实的角速度
            
            // 计算合速度（用于欺骗Autoware）
            double resultant_velocity = std::sqrt(vx_actual * vx_actual + vy_actual * vy_actual);
            
            // 计算等效转向角（将横向运动转换为等效转向）
            double equivalent_steering = 0.0;
            if (std::abs(vx_actual) > 0.01) {  // 避免除零
                // 简化的阿克曼几何关系：steering ≈ lateral_displacement / longitudinal_distance
                equivalent_steering = std::atan2(vy_actual, std::abs(vx_actual));
                // 限制转向角范围
                equivalent_steering = std::clamp(equivalent_steering, -0.5, 0.5);
            }
            
            // 填充阿克曼控制指令（欺骗Autoware）
            ackermann_cmd.longitudinal.velocity = resultant_velocity;  // 合速度
            ackermann_cmd.longitudinal.acceleration = 0.0;  // 简化处理
            ackermann_cmd.longitudinal.jerk = 0.0;
            ackermann_cmd.lateral.steering_tire_angle = equivalent_steering;  // 等效转向角
            ackermann_cmd.lateral.steering_tire_rotation_rate = 0.0;
            
            RCLCPP_DEBUG(this->get_logger(), "阿克曼转换: vx=%.3f, vy=%.3f -> vel=%.3f, steer=%.3f", 
                        vx_actual, vy_actual, resultant_velocity, equivalent_steering);
        } else {
            // 如果没有调试数据，使用MPC输出作为后备
            ackermann_cmd = mpc_cmd;
        }
        
        return ackermann_cmd;
    }

    /**
     * @brief 发布真正的麦克纳姆轮控制指令
     * 直接使用MPC计算出的[vx, vy, omega]控制麦克纳姆轮
     */
    void publishMecanumControl(const Float32MultiArray& debug_data) {
        if (debug_data.data.size() >= 6) {
            // 提取真实的控制输出
            double vx_mecanum = debug_data.data[3];  // 纵向速度
            double vy_mecanum = debug_data.data[4];  // 横向速度  
            double omega_mecanum = debug_data.data[5]; // 角速度
            
            // 发布Twist消息控制麦克纳姆轮
            Twist mecanum_twist;
            mecanum_twist.linear.x = vx_mecanum;
            mecanum_twist.linear.y = vy_mecanum;
            mecanum_twist.linear.z = 0.0;
            mecanum_twist.angular.x = 0.0;
            mecanum_twist.angular.y = 0.0;
            mecanum_twist.angular.z = omega_mecanum;
            
            pub_mecanum_twist_->publish(mecanum_twist);
            
            // 计算并发布轮速
            MPCTracker::ControlVector mpc_control(vx_mecanum, vy_mecanum, omega_mecanum);
            std::vector<double> wheel_speeds = mpc_tracker_->computeWheelSpeeds(mpc_control);
            Float32MultiArray wheel_msg;
            for (double speed : wheel_speeds) {
                wheel_msg.data.push_back(static_cast<float>(speed));
            }
            pub_wheel_speeds_->publish(wheel_msg);
            
            RCLCPP_DEBUG(this->get_logger(), "发布麦克纳姆轮控制: vx=%.3f, vy=%.3f, omega=%.3f", 
                        vx_mecanum, vy_mecanum, omega_mecanum);
        }
    }

    // 【修正】辅助函数参数类型也要改
    void publishAuxTopics(const ControlCmd& mpc_cmd, const Trajectory& pred_traj, const Float32MultiArray& debug) {
        (void)mpc_cmd; // 消除未使用警告

        // 发布预测轨迹
        Trajectory pred_traj_msg = pred_traj;
        pred_traj_msg.header.stamp = this->now();
        pred_traj_msg.header.frame_id = "map";
        pub_pred_traj_->publish(pred_traj_msg);

        // 发布调试数据
        pub_debug_->publish(debug);
    }

    std::unique_ptr<MPCTracker> mpc_tracker_;
    rclcpp::Subscription<Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<Trajectory>::SharedPtr sub_traj_;

    rclcpp::Publisher<ControlCmd>::SharedPtr pub_control_cmd_; // 用于欺骗Autoware的阿克曼指令
    rclcpp::Publisher<Twist>::SharedPtr pub_mecanum_twist_;    // 真正的麦克纳姆轮控制指令
    rclcpp::Publisher<Trajectory>::SharedPtr pub_pred_traj_;
    rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_debug_;
    rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_wheel_speeds_;

    rclcpp::TimerBase::SharedPtr timer_;
    std::mutex data_mtx_;
    Odometry current_odom_;
    Trajectory current_traj_;
    bool has_odom_ = false;
    bool has_traj_ = false;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPCControllerNode>());
    rclcpp::shutdown();
    return 0;
}