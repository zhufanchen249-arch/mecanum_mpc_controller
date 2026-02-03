#ifndef MPC_TRACKER_HPP
#define MPC_TRACKER_HPP

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <string>
// 【移除】不再引用 Autoware 内部接口，解决编译找不到文件的问题
// #include "autoware/mpc_lateral_controller/vehicle_model/vehicle_model_interface.hpp"

#include "autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_adapi_v1_msgs/msg/float32_multi_array_stamped.hpp"
#include "tf2/utils.h"
#include "rclcpp/rclcpp.hpp"

namespace autoware::mpc_lateral_controller {

// 【修改】移除继承 : public VehicleModelInterface
class MPCTracker {
public:
    // 类型别名提高可读性
    using StateVector = Eigen::Matrix<double, 6, 1>;  // 固定6维，6维状态向量 [x, y, theta, vx, vy, omega]
    using OutputVector = Eigen::Vector3d; // 3维输出向量 [x, y, theta]
    using ControlVector = Eigen::Vector3d; // 3维控制向量 [delta_vx, delta_vy, delta_omega]

    /**
     * @brief 构造函数（适配Autoware接口）
     * @param configFile YAML配置文件路径
     * @param wheelbase 车辆轴距
     */
    MPCTracker(const std::string& configFile, double wheelbase);

    /**
     * @brief 从YAML文件加载配置
     * @param configFile 配置文件路径
     * @return 是否加载成功
     */
    bool loadConfig(const std::string& configFile);

    /**
     * @brief 保存当前配置到YAML文件
     * @param configFile 保存路径
     * @return 是否保存成功
     */
    bool saveConfig(const std::string& configFile);

    // 设置和获取参数的接口
    void setPredictionHorizon(int N);
    void setSamplingTime(double dt);
    void setStateBounds(const StateVector& x_min, const StateVector& x_max);
    void setControlBounds(const ControlVector& u_min, const ControlVector& u_max);
    void setWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);
    void setReferenceTrajectory(const std::vector<StateVector>& ref_traj);
    void setReferenceControl(const std::vector<ControlVector>& ref_control);
    void setCurrentState(const StateVector& state);
    void setOutputMatrix(const Eigen::MatrixXd& C_tilde);

    // 获取预测轨迹（调试用）
    std::vector<OutputVector> getPredictedTrajectory() const;

    // 求解MPC问题
    bool solve(ControlVector& optimal_control);

    // 获取当前配置参数（用于保存配置）
    int getPredictionHorizon() const { return N_; }
    double getSamplingTime() const { return dt_; }
    StateVector getStateLowerBounds() const { return x_min_; }
    StateVector getStateUpperBounds() const { return x_max_; }
    ControlVector getControlLowerBounds() const { return u_min_; }
    ControlVector getControlUpperBounds() const { return u_max_; }
    Eigen::MatrixXd getQWeight() const { return Q_; }
    Eigen::MatrixXd getRWeight() const { return R_; }

    // 【修改】移除 override 关键字，因为不再继承基类
    bool calculateMPC(
        const autoware_auto_control_msgs::msg::AckermannLateralCommand & current_steer,
        const nav_msgs::msg::Odometry & current_kinematics,
        autoware_auto_control_msgs::msg::AckermannLateralCommand & output,
        autoware_auto_planning_msgs::msg::Trajectory & predicted_traj,
        autoware_adapi_v1_msgs::msg::Float32MultiArrayStamped & debug_values);

private:
    // MPC核心参数
    int N_; // 预测步长
    double dt_; // 采样时间
    double wheelbase_; // 车辆轴距（新增）


    // 麦克纳姆轮参数
    double wheel_radius_;       // 轮子半径
    double track_width_;        // 左右轮距
    Eigen::MatrixXd M_;         // 逆向运动学矩阵
    Eigen::MatrixXd M_inv_;     // 正向运动学矩阵（伪逆）

    // 约束
    StateVector x_min_; // 状态下界 [x, y, theta, vx, vy, omega]
    StateVector x_max_; // 状态上界
    ControlVector u_min_; // 控制输入下界 [delta_vx, delta_vy, delta_omega]
    ControlVector u_max_; // 控制输入上界

    StateVector current_state_; // 当前状态
    std::vector<StateVector> ref_traj_; // 参考轨迹
    std::vector<ControlVector> ref_control_; // 参考控制输入

    // 代价函数权重
    Eigen::MatrixXd Q_; // 状态误差权重矩阵
    Eigen::MatrixXd R_; // 控制输入权重矩阵
    Eigen::MatrixXd C_tilde_;// 输出矩阵

    // 求解器相关
    OsqpEigen::Solver solver_;
    bool solver_initialized_;
    std::vector<OutputVector> last_pred_traj_; // 最后一次预测轨迹

    // 辅助函数
    ControlVector saturateControl(const ControlVector& u) const;
    void buildAugmentedAB(const ControlVector& ref_u, double theta,
                         Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug);
    Eigen::MatrixXd buildPhiAug();
    Eigen::MatrixXd buildThetaAug();
    bool solveQP(ControlVector& optimal_control);

    // 初始化权重矩阵
    void initializeWeights();
    
    // 新增：轨迹转换工具函数（Autoware轨迹 -> 内部轨迹）
    bool convertAutowareTrajectoryToInternal(
        const autoware_auto_planning_msgs::msg::Trajectory& autoware_traj,
        std::vector<StateVector>& internal_traj);
    
    // 新增：麦克纳姆轮模型初始化
    void initializeMecanumModel();
    
    
    // 新增：运动学转换函数
    std::vector<double> computeWheelSpeeds(const ControlVector& u);
    Eigen::Vector3d computeTwist(const std::vector<double>& wheel_speeds);
};

} // namespace autoware::mpc_lateral_controller

#endif // MPC_TRACKER_HPP