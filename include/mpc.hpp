#ifndef MPC_TRACKER_HPP
#define MPC_TRACKER_HPP

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <string>

// 仅引用标准消息和 Autoware 核心消息
#include "autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"  // 调试用标准消息
#include "rclcpp/rclcpp.hpp"
#include "tf2/utils.h"

namespace autoware::mpc_lateral_controller {

// 这是一个独立的类，没有任何继承
class MPCTracker {
public:
    using StateVector = Eigen::Matrix<double, 6, 1>;
    using OutputVector = Eigen::Vector3d;
    using ControlVector = Eigen::Vector3d;

    MPCTracker(const std::string& configFile, double wheelbase);

    // 参数设置接口
    void setReferenceTrajectory(const std::vector<StateVector>& ref_traj);
    bool convertAutowareTrajectoryToInternal(
        const autoware_auto_planning_msgs::msg::Trajectory& autoware_traj,
        std::vector<StateVector>& internal_traj);

    // 核心计算接口
    bool calculateMPC(
        const autoware_auto_control_msgs::msg::AckermannLateralCommand & current_steer,
        const nav_msgs::msg::Odometry & current_kinematics,
        autoware_auto_control_msgs::msg::AckermannLateralCommand & output,
        autoware_auto_planning_msgs::msg::Trajectory & predicted_traj,
        std_msgs::msg::Float32MultiArray & debug_values); // 注意：这里用的是标准 std_msgs

    // 辅助功能
    std::vector<double> computeWheelSpeeds(const ControlVector& u);

private:
    bool loadConfig(const std::string& configFile);
    void initializeMecanumModel();
    void initializeWeights();
    bool solve(ControlVector& optimal_control);
    bool solveQP(ControlVector& optimal_control);
    
    // 成员变量
    int N_;
    double dt_;
    double wheelbase_;
    double wheel_radius_;
    double track_width_;

    Eigen::MatrixXd Q_, R_;
    Eigen::MatrixXd M_, M_inv_;
    
    StateVector x_min_, x_max_;
    ControlVector u_min_, u_max_;
    
    StateVector current_state_;
    std::vector<StateVector> ref_traj_;
    std::vector<ControlVector> ref_control_;
    
    OsqpEigen::Solver solver_;
    bool solver_initialized_;
    std::vector<OutputVector> last_pred_traj_;

    // 内部辅助函数
    ControlVector saturateControl(const ControlVector& u) const;
    void buildAugmentedAB(const ControlVector& ref_u, double theta,
                         Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug);
    Eigen::MatrixXd buildPhiAug();
    Eigen::MatrixXd buildThetaAug();
 };

} // namespace autoware::mpc_lateral_controller
#endif