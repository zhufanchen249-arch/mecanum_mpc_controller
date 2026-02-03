#include <memory>
#include <mutex>
#include <string>
#include <Eigen/Dense>
// ROS2核心头文件
#include "rclcpp/rclcpp.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
// 算法层头文件
#include "mpc.hpp"
// 消息类型头文件（无任何adapi引用，严格按要求引入）
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

// 简化命名空间，提升代码可读性
using namespace std::chrono_literals;
using autoware::mpc_lateral_controller::MPCTracker;
using LateralCmd = autoware_auto_control_msgs::msg::AckermannLateralCommand;
using ControlCmd = autoware_auto_control_msgs::msg::AckermannControlCommand;
using Trajectory = autoware_auto_planning_msgs::msg::Trajectory;
using Odometry = nav_msgs::msg::Odometry;
using Twist = geometry_msgs::msg::Twist;
using Float32MultiArray = std_msgs::msg::Float32MultiArray;

using std::placeholders::_1;

class MPCControllerNode : public rclcpp::Node {
public:
    MPCControllerNode() : Node("mpc_controller_mecanum") {
        // 1. 声明节点参数（带默认值，支持启动时重写）
        this->declare_parameter("config_file", "mecanum_car.yaml");
        this->declare_parameter("wheelbase", 0.3);
        this->declare_parameter("control_rate", 20.0);
        
        // 2. 获取参数并处理配置文件路径
        std::string config_file = this->get_parameter("config_file").as_string();
        double wheelbase = this->get_parameter("wheelbase").as_double();
        double control_rate = this->get_parameter("control_rate").as_double();

        // 自动拼接包路径：若未指定绝对路径，从包的param目录加载
        if (config_file.find("/") == std::string::npos) {
            std::string pkg_share = ament_index_cpp::get_package_share_directory("mecanum_mpc_controller");
            config_file = pkg_share + "/param/" + config_file;
            RCLCPP_INFO(this->get_logger(), "配置文件路径：%s", config_file.c_str());
        }

        // 3. 初始化MPC算法核心（与修正后的mpc.hpp构造函数匹配）
        try {
            mpc_tracker_ = std::make_unique<MPCTracker>(config_file, wheelbase);
            RCLCPP_INFO(this->get_logger(), "MPC跟踪器初始化成功！");
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "MPC跟踪器初始化失败：%s", e.what());
            rclcpp::shutdown();
            return;
        }

        // 4. 创建订阅器：里程计、参考轨迹（带互斥锁，保证数据线程安全）
        sub_odom_ = this->create_subscription<Odometry>(
            "input/odometry", 10, std::bind(&MPCControllerNode::onOdom, this, _1));
        sub_traj_ = this->create_subscription<Trajectory>(
            "input/trajectory", 10, std::bind(&MPCControllerNode::onTraj, this, _1));

        // 5. 创建发布者【核心：仅发布AckermannControlCommand】+ 辅助话题
        pub_control_cmd_ = this->create_publisher<ControlCmd>("output/control_cmd", 10); // Autoware标准控制指令
        pub_twist_ = this->create_publisher<Twist>("output/mecanum_twist", 10);         // 底盘原始Twist指令
        pub_pred_traj_ = this->create_publisher<Trajectory>("output/predicted_trajectory", 10); // MPC预测轨迹
        pub_debug_ = this->create_publisher<Float32MultiArray>("output/mpc_debug", 10); // MPC调试数据
        pub_wheel_speeds_ = this->create_publisher<Float32MultiArray>("output/mecanum_wheel_speeds", 10); // 四轮转速

        // 6. 创建控制定时器（按指定频率执行MPC计算，默认20Hz）
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / control_rate),
            std::bind(&MPCControllerNode::mpcControlLoop, this)
        );

        RCLCPP_INFO(this->get_logger(), "麦克纳姆轮MPC控制器节点启动成功！控制频率：%.1fHz", control_rate);
    }

private:
    // -------------------------- 回调函数：里程计订阅 --------------------------
    void onOdom(const Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mtx_); // 加锁保证数据一致性
        current_odom_ = *msg;
        has_odom_ = true;
    }

    // -------------------------- 回调函数：参考轨迹订阅 --------------------------
    void onTraj(const Trajectory::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mtx_);
        current_traj_ = *msg;
        has_traj_ = true;
    }

    // -------------------------- 核心控制循环：MPC计算+指令发布 --------------------------
    void mpcControlLoop() {
        // 前置检查：里程计和参考轨迹均已接收到，否则直接返回
        if (!has_odom_ || !has_traj_) {
            static int warn_count = 0;
            if (++warn_count % 100 == 0) { // 避免日志刷屏，每100次打印一次警告
                RCLCPP_WARN(this->get_logger(), "未接收到里程计或参考轨迹，等待中...");
            }
            return;
        }

        // 线程安全拷贝数据（减少锁持有时间）
        Odometry odom;
        Trajectory traj;
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            odom = current_odom_;
            traj = current_traj_;
        }

        // 1. 将Autoware标准轨迹转换为MPC算法内部6维状态轨迹【类型与mpc.hpp完全匹配】
        std::vector<MPCTracker::StateVector> inner_traj;
        if (!mpc_tracker_->convertAutowareTrajectoryToInternal(traj, inner_traj) || inner_traj.empty()) {
            RCLCPP_WARN(this->get_logger(), "轨迹转换失败或内部轨迹为空！");
            return;
        }

        // 2. 设置MPC参考轨迹和参考控制输入【类型与mpc.hpp完全匹配】
        if (!mpc_tracker_->setReferenceTrajectory(inner_traj)) {
            RCLCPP_WARN(this->get_logger(), "参考轨迹长度与MPC预测步长不匹配！");
            return;
        }
        // 参考控制初始化为0，MPC将基于此优化出最优控制
        std::vector<MPCTracker::ControlVector> ref_control(inner_traj.size(), MPCTracker::ControlVector::Zero());
        mpc_tracker_->setReferenceControl(ref_control);

        // 3. 初始化MPC计算所需变量【与mpc.hpp的calculateMPC接口严格匹配：LateralCmd】
        LateralCmd dummy_input;  // 算法层输入（预留，麦克纳姆轮无转向，置空即可）
        LateralCmd mpc_lateral_cmd; // 算法层输出（MPC求解的横向+纵向指令）
        Trajectory pred_traj_msg;   // MPC预测轨迹（可视化用）
        Float32MultiArray debug_msg;// MPC调试数据

        // 4. 执行核心MPC计算【调用修正后的mpc.cpp接口，无类型不匹配】
        if (!mpc_tracker_->calculateMPC(dummy_input, odom, mpc_lateral_cmd, pred_traj_msg, debug_msg)) {
            RCLCPP_WARN(this->get_logger(), "MPC控制指令求解失败！");
            return;
        }

        // 5. 算法层输出(LateralCmd)转换为Autoware标准控制指令(ControlCmd)【核心：对外仅发布此类型】
        ControlCmd autoware_control_cmd;
        // 填充时间戳和坐标系（关键，保证Autoware系统时间同步）
        autoware_control_cmd.stamp = this->now();
        autoware_control_cmd.header.frame_id = "map";
        // 复制MPC求解的纵向指令（速度/加速度）
        autoware_control_cmd.longitudinal = mpc_lateral_cmd.longitudinal;
        // 复制MPC求解的横向指令（麦克纳姆轮无转向，均为0）
        autoware_control_cmd.lateral = mpc_lateral_cmd.lateral;

        // 6. 发布Autoware标准控制指令【核心要求：仅发布AckermannControlCommand】
        pub_control_cmd_->publish(autoware_control_cmd);

        // 7. 发布辅助话题：底盘Twist、四轮转速、预测轨迹、调试数据（带鲁棒性判空）
        publishAuxTopics(mpc_lateral_cmd, pred_traj_msg, debug_msg);
    }

    // -------------------------- 辅助函数：发布底盘/可视化/调试话题 --------------------------
    void publishAuxTopics(const LateralCmd& mpc_cmd, const Trajectory& pred_traj, const Float32MultiArray& debug) {
        // 发布底盘原始Twist指令（x/y线速度、z角速度，适配麦克纳姆轮全向运动）
        if (debug.data.size() >= 6) {
            Twist twist;
            twist.linear.x = debug.data[3];  // MPC求解的x方向速度
            twist.linear.y = debug.data[4];  // MPC求解的y方向速度
            twist.angular.z = debug.data[5]; // MPC求解的角速度
            pub_twist_->publish(twist);

            // 发布麦克纳姆轮四轮转速（逆向运动学计算）
            MPCTracker::ControlVector mpc_control(debug.data[3], debug.data[4], debug.data[5]);
            std::vector<double> wheel_speeds = mpc_tracker_->computeWheelSpeeds(mpc_control);
            Float32MultiArray wheel_msg;
            for (double speed : wheel_speeds) {
                wheel_msg.data.push_back(static_cast<float>(speed));
            }
            pub_wheel_speeds_->publish(wheel_msg);
        }

        // 发布MPC预测轨迹（填充header，适配Rviz可视化）
        Trajectory pred_traj_msg = pred_traj;
        pred_traj_msg.header.stamp = this->now();
        pred_traj_msg.header.frame_id = "map";
        pub_pred_traj_->publish(pred_traj_msg);

        // 发布MPC调试数据
        pub_debug_->publish(debug);
    }

    // -------------------------- 节点成员变量 --------------------------
    std::unique_ptr<MPCTracker> mpc_tracker_;          // MPC算法核心实例
    rclcpp::Subscription<Odometry>::SharedPtr sub_odom_;    // 里程计订阅器
    rclcpp::Subscription<Trajectory>::SharedPtr sub_traj_;  // 参考轨迹订阅器

    // 核心发布者【仅发布AckermannControlCommand，满足用户要求】
    rclcpp::Publisher<ControlCmd>::SharedPtr pub_control_cmd_;
    // 辅助发布者（底盘/可视化/调试）
    rclcpp::Publisher<Twist>::SharedPtr pub_twist_;
    rclcpp::Publisher<Trajectory>::SharedPtr pub_pred_traj_;
    rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_debug_;
    rclcpp::Publisher<Float32MultiArray>::SharedPtr pub_wheel_speeds_;

    rclcpp::TimerBase::SharedPtr timer_;  // 控制循环定时器
    std::mutex data_mtx_;                 // 数据互斥锁（保证线程安全）
    Odometry current_odom_;               // 最新里程计数据
    Trajectory current_traj_;             // 最新参考轨迹数据
    bool has_odom_ = false;               // 里程计数据就绪标志
    bool has_traj_ = false;               // 参考轨迹数据就绪标志
};

// -------------------------- 主函数：节点入口 --------------------------
int main(int argc, char **argv) {
    // ROS2初始化+节点自旋+退出
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPCControllerNode>());
    rclcpp::shutdown();
    return 0;
}