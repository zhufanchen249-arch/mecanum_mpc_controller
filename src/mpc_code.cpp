#include <memory>
#include <mutex>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "mpc.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp"
#include "autoware_adapi_v1_msgs/msg/float32_multi_array_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"  // 添加这一行

using namespace std::chrono_literals;
using std::placeholders::_1;

// 麦克纳姆轮MPC控制器节点
class MPCControllerNode : public rclcpp::Node
{
public:
    MPCControllerNode() : Node("mpc_controller_mecanum")
    {
        // 声明参数
        this->declare_parameter("config_file", "param/mecanum_car.yaml");
        this->declare_parameter("wheelbase", 0.3);
        this->declare_parameter("control_rate", 10.0);
        this->declare_parameter("publish_wheel_speeds", true);

        // 获取参数
        std::string config_file;
        double wheelbase;
        double control_rate;
        bool publish_wheel_speeds;

        this->get_parameter("config_file", config_file);
        this->get_parameter("wheelbase", wheelbase);
        this->get_parameter("control_rate", control_rate);
        this->get_parameter("publish_wheel_speeds", publish_wheel_speeds);

           // 处理配置文件路径
       if (config_file.find("/") == std::string::npos) {  // 如果是相对路径
           config_file = ament_index_cpp::get_package_share_directory("mecanum_mpc_controller") 
                + "/param/" + config_file;
        }   

        // 打印初始化信息
        RCLCPP_INFO(this->get_logger(), "=== 麦克纳姆轮MPC控制器初始化 ===");
        RCLCPP_INFO(this->get_logger(), "配置文件：%s", config_file.c_str());
        RCLCPP_INFO(this->get_logger(), "轮距：%.3f m", wheelbase);
        RCLCPP_INFO(this->get_logger(), "控制频率：%.1f Hz | 轮速发布：%s", 
                    control_rate, publish_wheel_speeds ? "启用" : "禁用");

        // 创建MPC跟踪器（修正：只传入两个参数）
        mpc_tracker_ = std::make_unique<MPCTracker>(config_file, wheelbase);  // 修正：移除wheel_radius参数

        // 创建订阅者
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "input/odometry", 10,
            std::bind(&MPCControllerNode::odometryCallback, this, _1));

        trajectory_sub_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
            "input/trajectory", 10,
            std::bind(&MPCControllerNode::trajectoryCallback, this, _1));

        // 创建发布者
        control_cmd_pub_ = this->create_publisher<autoware_auto_control_msgs::msg::AckermannLateralCommand>(
            "output/ackermann_cmd", 10);

        twist_cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "output/mecanum_twist", 10);

        predicted_trajectory_pub_ = this->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
            "output/predicted_trajectory", 10);

        debug_values_pub_ = this->create_publisher<autoware_adapi_v1_msgs::msg::Float32MultiArrayStamped>(
            "output/mpc_debug", 10);

        if (publish_wheel_speeds) {
            wheel_speeds_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
                "output/mecanum_wheel_speeds", 10);
            RCLCPP_INFO(this->get_logger(), "轮速发布话题：output/mecanum_wheel_speeds");
        }

        // 创建控制定时器
        double timer_period = 1.0 / control_rate;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(timer_period),
            std::bind(&MPCControllerNode::controlLoop, this));

        RCLCPP_INFO(this->get_logger(), "=== 麦克纳姆轮MPC控制器初始化完成 ===");
    }

private:
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        current_odometry_ = *msg;
        odom_received_ = true;
    }

    void trajectoryCallback(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        current_trajectory_ = *msg;
        trajectory_received_ = true;
    }

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

        // 轨迹格式转换
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


        // 设置参考轨迹
        mpc_tracker_->setReferenceTrajectory(internal_traj);

        // 执行MPC计算
        bool mpc_success = mpc_tracker_->calculateMPC(
            dummy_steer,
            current_odom,
            ackermann_output,
            predicted_trajectory,
            debug_values
        );

        // MPC计算成功：发布控制指令
        if (mpc_success) {
            // 发布备用阿克曼指令
            ackermann_output.stamp = this->now();
            control_cmd_pub_->publish(ackermann_output);

        // 发布全向速度指令
        geometry_msgs::msg::Twist mecanum_twist;
        if (debug_values.data.size() >= 3) {
            mecanum_twist.linear.x = debug_values.data[0];
            mecanum_twist.linear.y = debug_values.data[1];
            mecanum_twist.angular.z = debug_values.data[2];
            mecanum_twist.header.stamp = this->now();
            mecanum_twist.header.frame_id = "base_link";
            twist_cmd_pub_->publish(mecanum_twist);
            RCLCPP_DEBUG(this->get_logger(), 
                        "全向控制指令：vx=%.2f m/s, vy=%.2f m/s, omega=%.2f rad/s",
                        mecanum_twist.linear.x, mecanum_twist.linear.y, mecanum_twist.angular.z);
        } else {
            RCLCPP_WARN(this->get_logger(), "调试数据不完整！无法发布全向速度指令");
             }

            // 发布轮速指令
            if (wheel_speeds_pub_) {
                autoware::mpc_lateral_controller::MPCTracker::ControlVector control_vec(
                    debug_values.data[0], debug_values.data[1], debug_values.data[2]);
                auto wheel_speeds = mpc_tracker_->computeWheelSpeeds(control_vec);
                if (wheel_speeds.size() == 4) {
                    std_msgs::msg::Float32MultiArray wheel_speeds_msg;
                    wheel_speeds_msg.data = {
                        static_cast<float>(wheel_speeds[0]),
                        static_cast<float>(wheel_speeds[1]),
                        static_cast<float>(wheel_speeds[2]),
                        static_cast<float>(wheel_speeds[3])
                    };
                    wheel_speeds_pub_->publish(wheel_speeds_msg);
                    RCLCPP_DEBUG(this->get_logger(), 
                                "轮速指令（rad/s）：FL=%.2f, FR=%.2f, RL=%.2f, RR=%.2f",
                                wheel_speeds[0], wheel_speeds[1], wheel_speeds[2], wheel_speeds[3]);
                } else {
                    RCLCPP_WARN(this->get_logger(), "轮速计算结果无效！需返回4个轮速值");
                }
            }

            // 发布预测轨迹
            predicted_trajectory.header.stamp = this->now();
            predicted_trajectory.header.frame_id = "map";
            predicted_trajectory_pub_->publish(predicted_trajectory);

            // 发布调试数据
            debug_values.header.stamp = this->now();
            debug_values_pub_->publish(debug_values);

        } else {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                "MPC计算失败！");
        }
    }

    // 成员变量
    std::unique_ptr<autoware::mpc_lateral_controller::MPCTracker> mpc_tracker_;

    // 订阅者
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr trajectory_sub_;

    // 发布者
    rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannLateralCommand>::SharedPtr control_cmd_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_cmd_pub_;
    rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr predicted_trajectory_pub_;
    rclcpp::Publisher<autoware_adapi_v1_msgs::msg::Float32MultiArrayStamped>::SharedPtr debug_values_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr wheel_speeds_pub_;

    // 定时器
    rclcpp::TimerBase::SharedPtr timer_;

    // 数据存储与线程安全保护
    nav_msgs::msg::Odometry current_odometry_;
    autoware_auto_planning_msgs::msg::Trajectory current_trajectory_;
    std::mutex odom_mutex_;
    std::mutex trajectory_mutex_;
    bool odom_received_ = false;
    bool trajectory_received_ = false;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MPCControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}