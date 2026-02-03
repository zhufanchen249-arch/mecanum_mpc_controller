#ifndef MPC_TRACKER_HPP
#define MPC_TRACKER_HPP

// 基础算法库头文件
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <string>
#include <cmath>

// 【严格按要求引入】核心正确头文件
#include "autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "tf2/utils.h"

namespace autoware::mpc_lateral_controller {

class MPCTracker {
public:
    // 类型定义【与mpc.cpp完全一致】，避免类型不匹配
    using StateVector = Eigen::VectorXd;
    using ControlVector = Eigen::Vector3d;
    using OutputVector = Eigen::Vector3d;

    /**
     * @brief 构造函数
     * @param configFile 配置文件路径
     * @param wheelbase 机器人轮距（麦克纳姆轮关键参数）
     */
    MPCTracker(const std::string& configFile, double wheelbase);

    /**
     * @brief 核心MPC计算接口【参数与cpp实现1:1匹配，无错误】
     * @param current_steer 当前转向指令（预留，麦克纳姆轮无转向）
     * @param current_kinematics 当前机器人里程计状态
     * @param output 输出的MPC控制指令
     * @param predicted_traj MPC预测轨迹（用于Rviz可视化）
     * @param debug_values 调试数据（std_msgs::Float32MultiArray，无Stamped）
     * @return 求解成功返回true，失败返回false
     */
    bool calculateMPC(
        const autoware_auto_control_msgs::msg::AckermannLateralCommand & current_steer,
        const nav_msgs::msg::Odometry & current_kinematics,
        autoware_auto_control_msgs::msg::AckermannLateralCommand & output,
        autoware_auto_planning_msgs::msg::Trajectory & predicted_traj,
        std_msgs::msg::Float32MultiArray & debug_values);

    /**
     * @brief 将Autoware标准轨迹转换为MPC内部轨迹格式
     * @param autoware_traj Autoware原生轨迹消息
     * @param internal_traj 转换后的MPC内部6维状态轨迹
     * @return 转换成功返回true，失败返回false
     */
    bool convertAutowareTrajectoryToInternal(
        const autoware_auto_planning_msgs::msg::Trajectory& autoware_traj,
        std::vector<StateVector>& internal_traj);

    /**
     * @brief 设置MPC参考轨迹
     * @param ref_traj 6维状态参考轨迹
     * @return 设置成功返回true，失败返回false
     */
    bool setReferenceTrajectory(const std::vector<StateVector>& ref_traj);

    /**
     * @brief 设置MPC参考控制输入
     * @param ref_control 3维控制参考输入
     * @return 设置成功返回true，失败返回false
     */
    bool setReferenceControl(const std::vector<ControlVector>& ref_control);

    /**
     * @brief 麦克纳姆轮逆向运动学计算
     * @param u MPC求解的控制指令[vx, vy, omega]
     * @return 四轮转速[前左, 前右, 后左, 后右]
     */
    std::vector<double> computeWheelSpeeds(const ControlVector& u);

private:
    // -------------------------- 核心参数（与cpp顺序一致，避免reorder警告）--------------------------
    int N_;                  // MPC预测步长
    double dt_;              // MPC采样时间
    double wheelbase_;       // 机器人轮距（前后轮中心距离）
    double wheel_radius_;    // 麦克纳姆轮半径
    double track_width_;     // 机器人轴距（左右轮中心距离）

    ControlVector u_min_;    // 控制输入下界[vx_min, vy_min, omega_min]
    ControlVector u_max_;    // 控制输入上界[vx_max, vy_max, omega_max]
    Eigen::VectorXd x_min_;  // 状态变量下界[6维]
    Eigen::VectorXd x_max_;  // 状态变量上界[6维]
    Eigen::VectorXd current_state_; // 机器人当前6维状态[x, y, yaw, vx, vy, omega]

    // -------------------------- 矩阵与求解器（与cpp顺序一致）--------------------------
    Eigen::MatrixXd Q_;      // 状态误差权重矩阵
    Eigen::MatrixXd R_;      // 控制输入权重矩阵
    Eigen::MatrixXd M_;      // 麦克纳姆轮逆向运动学矩阵[4x3]
    Eigen::MatrixXd M_inv_;  // 麦克纳姆轮正向运动学伪逆矩阵[3x4]
    Eigen::MatrixXd C_tilde_;// 输出映射矩阵（状态->观测）
    OsqpEigen::Solver solver_; // OSQP-QP求解器
    bool solver_initialized_; // 求解器初始化标志

    // -------------------------- 轨迹缓存（与cpp顺序一致）--------------------------
    std::vector<OutputVector> last_pred_traj_; // 上一次MPC预测轨迹（用于可视化）
    std::vector<StateVector> ref_traj_;        // MPC参考轨迹缓存
    std::vector<ControlVector> ref_control_;   // MPC参考控制缓存

    // -------------------------- 内部核心函数（全部与cpp实现对应，无冗余声明）--------------------------
    /**
     * @brief 初始化MPC权重矩阵Q和R
     */
    void initializeWeights();

    /**
     * @brief 加载yaml配置文件
     * @param configFile 配置文件路径
     * @return 加载成功返回true，失败返回false
     */
    bool loadConfig(const std::string& configFile);

    /**
     * @brief 初始化麦克纳姆轮运动学模型
     */
    void initializeMecanumModel();

    /**
     * @brief 控制输入饱和处理（防止超出硬件限制）
     * @param u 未饱和的控制输入
     * @return 饱和后的控制输入
     */
    ControlVector saturateControl(const ControlVector& u) const;

    /**
     * @brief 构建MPC增广状态空间矩阵A和B
     * @param ref_u 参考控制输入
     * @param theta 当前偏航角
     * @param A_aug 输出的增广状态矩阵
     * @param B_aug 输出的增广控制矩阵
     */
    void buildAugmentedAB(const ControlVector& ref_u, double theta,
                          Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug);

    /**
     * @brief 构建MPC Phi矩阵（状态->输出映射）
     * @return 构建完成的Phi矩阵
     */
    Eigen::MatrixXd buildPhiAug();

    /**
     * @brief 构建MPC Theta矩阵（控制->输出映射）
     * @return 构建完成的Theta矩阵
     */
    Eigen::MatrixXd buildThetaAug();

    /**
     * @brief MPC求解主函数
     * @param optimal_control 输出的最优控制指令
     * @return 求解成功返回true，失败返回false
     */
    bool solve(ControlVector& optimal_control);

    /**
     * @brief 求解QP二次规划问题（OSQP核心调用）
     * @param optimal_control 输出的最优控制指令
     * @return 求解成功返回true，失败返回false
     */
    bool solveQP(ControlVector& optimal_control);
};

} // namespace autoware::mpc_lateral_controller

#endif // MPC_TRACKER_HPP