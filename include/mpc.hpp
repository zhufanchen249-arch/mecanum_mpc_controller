#ifndef MPC_TRACKER_HPP
#define MPC_TRACKER_HPP

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <string>

#include "autoware_control_msgs/msg/control.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "autoware_planning_msgs/msg/trajectory.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Quaternion.h"

namespace autoware::mpc_lateral_controller {

class MPCTracker {
public:
    using StateVector = Eigen::Matrix<double, 6, 1>;
    using OutputVector = Eigen::Vector3d;
    using ControlVector = Eigen::Vector3d;

    MPCTracker(const std::string& configFile, double wheelbase);

    bool loadConfig(const std::string& configFile);
    bool saveConfig(const std::string& configFile);

    void setPredictionHorizon(int N);
    void setSamplingTime(double dt);
    void setStateBounds(const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max);
    void setControlBounds(const ControlVector& u_min, const ControlVector& u_max);
    void setWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);
    
    // ✅ 这三个函数必须声明
    bool convertAutowareTrajectoryToInternal(
        const autoware_planning_msgs::msg::Trajectory& autoware_traj,
        std::vector<Eigen::Matrix<double, 6, 1>>& internal_traj);
    
    bool setReferenceTrajectory(const std::vector<Eigen::Matrix<double, 6, 1>>& ref_traj);
    bool setReferenceControl(const std::vector<ControlVector>& ref_control);
    
    void setCurrentState(const StateVector& state);

    bool calculateMPC(
        const autoware_control_msgs::msg::Control& current_control,
        const nav_msgs::msg::Odometry& current_kinematics,
        autoware_control_msgs::msg::Control& output,
        autoware_planning_msgs::msg::Trajectory& predicted_traj,
        std_msgs::msg::Float32MultiArray& debug_values);

    std::vector<double> computeWheelSpeeds(const ControlVector& u);
    std::vector<OutputVector> getPredictedTrajectory() const;

private:
    void initializeMecanumModel();
    void initializeWeights();
    bool solve(ControlVector& optimal_control);
    bool solveQP(ControlVector& optimal_control);
    
    int N_;
    double dt_;
    double wheelbase_;
    double wheel_radius_;
    double track_width_;

    ControlVector u_min_;
    ControlVector u_max_;
    Eigen::VectorXd x_min_;
    Eigen::VectorXd x_max_;
    Eigen::VectorXd current_state_;

    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd M_;
    Eigen::MatrixXd M_inv_;
    Eigen::MatrixXd C_tilde_;

    bool solver_initialized_;
    std::vector<OutputVector> last_pred_traj_;
    std::vector<Eigen::Matrix<double, 6, 1>> ref_traj_;
    std::vector<ControlVector> ref_control_;
    
    OsqpEigen::Solver solver_;

    ControlVector saturateControl(const ControlVector& u) const;
    void buildAugmentedAB(const ControlVector& ref_u, double theta,
                         Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug);
    Eigen::MatrixXd buildPhiAug();
    Eigen::MatrixXd buildThetaAug();
};

} // namespace autoware::mpc_lateral_controller
#endif