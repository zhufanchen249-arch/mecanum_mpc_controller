#include <memory>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
#include "mpc.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

using std::placeholders::_1;

class MPCControllerNode : public rclcpp::Node {
public:
    MPCControllerNode() : Node("mpc_controller_mecanum") {
        // 1. 加载参数
        this->declare_parameter("config_file", "param/mecanum_car.yaml");
        this->declare_parameter("wheelbase", 0.3);
        this->declare_parameter("control_rate", 20.0);
        
        std::string config_file = this->get_parameter("config_file").as_string();
        double wheelbase = this->get_parameter("wheelbase").as_double();
        double rate = this->get_parameter("control_rate").as_double();

        if (config_file.find("/") == std::string::npos) {
             config_file = ament_index_cpp::get_package_share_directory("mecanum_mpc_controller") + "/param/" + config_file;
        }

        // 2. 初始化 MPC 核心
        mpc_tracker_ = std::make_unique<autoware::mpc_lateral_controller::MPCTracker>(config_file, wheelbase);

        // 3. 订阅话题 (保持 Autoware 接口)
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "input/odometry", 10, std::bind(&MPCControllerNode::onOdom, this, _1));
        sub_traj_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
            "input/trajectory", 10, std::bind(&MPCControllerNode::onTraj, this, _1));

        // 4. 发布话题
        pub_cmd_ = this->create_publisher<autoware_auto_control_msgs::msg::AckermannLateralCommand>("output/ackermann_cmd", 10);
        pub_twist_ = this->create_publisher<geometry_msgs::msg::Twist>("output/mecanum_twist", 10);
        pub_pred_traj_ = this->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("output/predicted_trajectory", 10);
        
        // 关键：使用标准消息发布调试数据
        pub_debug_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("output/mpc_debug", 10);
        pub_wheels_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("output/mecanum_wheel_speeds", 10);

        // 5. 定时器
        timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0/rate), std::bind(&MPCControllerNode::loop, this));
    }

private:
    void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        odom_ = *msg;
        has_odom_ = true;
    }

    void onTraj(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        traj_ = *msg;
        has_traj_ = true;
    }

    void loop() {
        if (!has_odom_ || !has_traj_) return;

        // 准备数据
        nav_msgs::msg::Odometry current_odom;
        autoware_auto_planning_msgs::msg::Trajectory current_traj;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            current_odom = odom_;
            current_traj = traj_;
        }

        // 转换轨迹
        std::vector<autoware::mpc_lateral_controller::MPCTracker::StateVector> inner_traj;
        if (!mpc_tracker_->convertAutowareTrajectoryToInternal(current_traj, inner_traj)) return;
        mpc_tracker_->setReferenceTrajectory(inner_traj);

        // 准备输出容器
        autoware_auto_control_msgs::msg::AckermannLateralCommand cmd;
        autoware_auto_control_msgs::msg::AckermannLateralCommand dummy_in; // MPC接口需要但这里用不到
        autoware_auto_planning_msgs::msg::Trajectory pred_traj_msg;
        std_msgs::msg::Float32MultiArray debug_msg;

        // 计算！
        bool success = mpc_tracker_->calculateMPC(dummy_in, current_odom, cmd, pred_traj_msg, debug_msg);

        if (success) {
            // 发布 Ackermann 指令 (给车辆接口)
            cmd.stamp = this->now();
            pub_cmd_->publish(cmd);

            // 发布 Twist 指令 (全向移动)
            if (debug_msg.data.size() >= 6) {
                geometry_msgs::msg::Twist twist;
                twist.linear.x = debug_msg.data[3]; // cmd_vx
                twist.linear.y = debug_msg.data[4]; // cmd_vy
                twist.angular.z = debug_msg.data[5]; // cmd_omega
                pub_twist_->publish(twist);

                // 计算并发布轮速
                auto wheels = mpc_tracker_->computeWheelSpeeds({debug_msg.data[3], debug_msg.data[4], debug_msg.data[5]});
                std_msgs::msg::Float32MultiArray wheel_msg;
                for(auto w : wheels) wheel_msg.data.push_back(w);
                pub_wheels_->publish(wheel_msg);
            }

            // 发布调试轨迹
            pred_traj_msg.header.stamp = this->now();
            pred_traj_msg.header.frame_id = "map";
            pub_pred_traj_->publish(pred_traj_msg);

            // 发布调试数据
            pub_debug_->publish(debug_msg);
        }
    }

    std::unique_ptr<autoware::mpc_lateral_controller::MPCTracker> mpc_tracker_;
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr sub_traj_;
    
    rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannLateralCommand>::SharedPtr pub_cmd_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_twist_;
    rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr pub_pred_traj_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_debug_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_wheels_;

    rclcpp::TimerBase::SharedPtr timer_;
    std::mutex mtx_;
    nav_msgs::msg::Odometry odom_;
    autoware_auto_planning_msgs::msg::Trajectory traj_;
    bool has_odom_ = false;
    bool has_traj_ = false;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPCControllerNode>());
    rclcpp::shutdown();
    return 0;
}