#include "mpc.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm> 
#include <tf2/LinearMath/Quaternion.h>

namespace autoware::mpc_lateral_controller {

// 在 mpc.cpp 开头添加类型别名（方便修改）
using StateVector = Eigen::Matrix<double, 6, 1>;
using OutputVector = Eigen::Vector3d;
using ControlVector = Eigen::Vector3d;

// 【必须】初始化列表顺序匹配头文件
MPCTracker::MPCTracker(const std::string& configFile, double wheelbase)
    : N_(20), dt_(0.05), wheelbase_(wheelbase),
      wheel_radius_(0.05), track_width_(0.5),
      u_min_(ControlVector::Zero()), u_max_(ControlVector::Zero()),
      x_min_(Eigen::VectorXd::Zero(6)), x_max_(Eigen::VectorXd::Zero(6)),
      current_state_(Eigen::VectorXd::Zero(6)),
      Q_(Eigen::MatrixXd::Zero(60, 60)), R_(Eigen::MatrixXd::Zero(60, 60)),
      M_(Eigen::MatrixXd::Zero(4, 3)), M_inv_(Eigen::MatrixXd::Zero(3, 4)),
      C_tilde_(Eigen::MatrixXd::Identity(3, 6)),
      // ✅ 按照 mpc.hpp 中的声明顺序排列
      solver_initialized_(false), 
      last_pred_traj_(std::vector<OutputVector>()),
      ref_traj_(std::vector<Eigen::Matrix<double, 6, 1>>()),  
      ref_control_(std::vector<ControlVector>()),
      solver_(OsqpEigen::Solver())
{
    initializeWeights();
    // 默认参数
    x_min_ << -10.0, -5.0, -M_PI, -2.0, -2.0, -2.0;
    x_max_ << 10.0, 5.0, M_PI, 2.0, 2.0, 2.0;
    u_min_ << -1.0, -1.0, -1.5;
    u_max_ << 1.0, 1.0, 1.5;

    if (!loadConfig(configFile)) {
        std::cerr << "[MPCTracker] Config load failed, using defaults." << std::endl;
    }
    initializeMecanumModel();
    
    solver_.settings()->setVerbosity(false);
    solver_.settings()->setWarmStart(true);
}

void MPCTracker::initializeWeights() {
    Q_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
    R_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
    for (int i = 0; i < N_; ++i) {
        Q_.block<3, 3>(i * 3, i * 3) = 10.0 * Eigen::MatrixXd::Identity(3, 3);
        R_.block<3, 3>(i * 3, i * 3) = 0.1 * Eigen::MatrixXd::Identity(3, 3);
    }
}

bool MPCTracker::loadConfig(const std::string& configFile) {
    try {
        YAML::Node config = YAML::LoadFile(configFile);
        
        if (config["prediction_horizon"]) N_ = config["prediction_horizon"].as<int>();
        if (config["sampling_time"]) dt_ = config["sampling_time"].as<double>();

        if (config["state_constraints"]) {
            YAML::Node sc = config["state_constraints"];
            if (sc["x_min"] && sc["x_max"]) {
                auto x_min_vec = sc["x_min"].as<std::vector<double>>();
                auto x_max_vec = sc["x_max"].as<std::vector<double>>();
                for(int i=0; i<6; ++i) { x_min_(i) = x_min_vec[i]; x_max_(i) = x_max_vec[i]; }
            }
        }

        if (config["control_constraints"]) {
            YAML::Node cc = config["control_constraints"];
            if (cc["u_min"] && cc["u_max"]) {
                auto u_min_vec = cc["u_min"].as<std::vector<double>>();
                auto u_max_vec = cc["u_max"].as<std::vector<double>>();
                for(int i=0; i<3; ++i) { u_min_(i) = u_min_vec[i]; u_max_(i) = u_max_vec[i]; }
            }
        }

        if (config["weights"]) {
            YAML::Node w = config["weights"];
            double q = w["Q"].as<double>(10.0);
            double r = w["R"].as<double>(0.1);
            Q_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
            R_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
            for (int i = 0; i < N_; ++i) {
                Q_.block<3, 3>(i * 3, i * 3) = q * Eigen::MatrixXd::Identity(3, 3);
                R_.block<3, 3>(i * 3, i * 3) = r * Eigen::MatrixXd::Identity(3, 3);
            }
        } else {
            initializeWeights();
        }
        
        if (config["mecanum_wheel"]) {
            YAML::Node mw = config["mecanum_wheel"];
            wheel_radius_ = mw["radius"].as<double>(wheel_radius_);
            track_width_ = mw["track_width"].as<double>(track_width_);
            if (mw["wheelbase"]) wheelbase_ = mw["wheelbase"].as<double>(wheelbase_);
        }
        
        initializeMecanumModel();
        solver_initialized_ = false;
        return true;
    } catch (...) {
        return false;
    }
}

bool MPCTracker::saveConfig(const std::string& configFile) {
    (void)configFile;
    RCLCPP_WARN(rclcpp::get_logger("mpc"), "saveConfig is not implemented.");
    return false;
}

void MPCTracker::initializeMecanumModel() {
    double L = wheelbase_ / 2.0;
    double W = track_width_ / 2.0;
    M_.resize(4, 3);
    M_ << 1, -1, -(L + W), 1, 1, (L + W), 1, 1, -(L + W), 1, -1, (L + W);
    M_ /= wheel_radius_;
    M_inv_ = M_.completeOrthogonalDecomposition().pseudoInverse();
}

void MPCTracker::setPredictionHorizon(int N) { if (N > 0) N_ = N; }
void MPCTracker::setSamplingTime(double dt) { if (dt > 0) dt_ = dt; }
void MPCTracker::setStateBounds(const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max) { x_min_ = x_min; x_max_ = x_max; }
void MPCTracker::setControlBounds(const ControlVector& u_min, const ControlVector& u_max) { u_min_ = u_min; u_max_ = u_max; }
void MPCTracker::setWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) { Q_ = Q; R_ = R; }
void MPCTracker::setCurrentState(const StateVector& state) { 
    current_state_ = state;
}

std::vector<double> MPCTracker::computeWheelSpeeds(const ControlVector& u) {
    Eigen::Vector4d ws = M_ * u;
    return {ws(0), ws(1), ws(2), ws(3)};
}

std::vector<MPCTracker::OutputVector> MPCTracker::getPredictedTrajectory() const {
    return last_pred_traj_;
}

MPCTracker::ControlVector MPCTracker::saturateControl(const ControlVector& u) const {
    ControlVector out;
    out(0) = std::clamp(u(0), u_min_(0), u_max_(0));
    out(1) = std::clamp(u(1), u_min_(1), u_max_(1));
    out(2) = std::clamp(u(2), u_min_(2), u_max_(2));
    return out;
}

void MPCTracker::buildAugmentedAB(const ControlVector& ref_u, double theta,
                                  Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug) {
    (void)ref_u; 
    const double T = dt_;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    
    A_aug = Eigen::MatrixXd::Identity(6, 6);
    B_aug = Eigen::MatrixXd::Zero(6, 3);
    B_aug(0, 0) = c * T; B_aug(0, 1) = -s * T;
    B_aug(1, 0) = s * T; B_aug(1, 1) = c * T;
    B_aug(2, 2) = T;
    B_aug(3, 0) = 1; B_aug(4, 1) = 1; B_aug(5, 2) = 1;
}

Eigen::MatrixXd MPCTracker::buildPhiAug() {
    const int n = 6;
    const int p = 3;
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(p * N_, n);
    Eigen::MatrixXd A_aug, B_aug;
    Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd C_tilde = Eigen::MatrixXd::Zero(p, n);
    C_tilde.block(0, 0, p, p) = Eigen::MatrixXd::Identity(p, p);

    for (int i = 0; i < N_; ++i) {
        buildAugmentedAB(ref_control_[i], ref_traj_[i](2), A_aug, B_aug);
        if (i > 0) A_power = A_aug * A_power;
        Phi.block(i * p, 0, p, n) = C_tilde * A_power;
    }
    return Phi;
}

Eigen::MatrixXd MPCTracker::buildThetaAug() {
    const int n = 6;
    const int m = 3;
    const int p = 3;
    Eigen::MatrixXd Theta = Eigen::MatrixXd::Zero(p * N_, m * N_);
    std::vector<Eigen::MatrixXd> A_list(N_), B_list(N_);

    for (int i = 0; i < N_; ++i) {
        buildAugmentedAB(ref_control_[i], ref_traj_[i](2), A_list[i], B_list[i]);
    }
    Eigen::MatrixXd C_tilde = Eigen::MatrixXd::Zero(p, n);
    C_tilde.block(0, 0, p, p) = Eigen::MatrixXd::Identity(p, p);

    for (int i = 0; i < N_; ++i) {
        Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);
        for (int j = 0; j <= i; ++j) {
            if (j > 0) A_power = A_list[i - j] * A_power; 
            Theta.block(i * p, j * m, p, m) = C_tilde * A_power * B_list[j];
        }
    }
    return Theta;
}
bool MPCTracker::solveQP(ControlVector& optimal_control) {
    if (ref_traj_.empty() || (int)ref_traj_.size() < N_) return false;

    const int control_dim = 3; 
    const int var_count = control_dim * N_;
    
    Eigen::VectorXd xi0 = current_state_ - ref_traj_[0];
    Eigen::VectorXd Y_ref(3 * N_);
    for (int i = 0; i < N_; ++i) Y_ref.segment<3>(i * 3) = ref_traj_[i].head<3>();

    Eigen::MatrixXd Phi = buildPhiAug();
    Eigen::MatrixXd Theta = buildThetaAug();
    
    Eigen::SparseMatrix<double> H = (Theta.transpose() * Q_ * Theta + R_).sparseView();
    Eigen::VectorXd g = Theta.transpose() * Q_ * (Phi * xi0 - Y_ref);

    // 构造约束矩阵
    Eigen::SparseMatrix<double> A(var_count, var_count);
    A.setIdentity(); 
    
    Eigen::VectorXd l(var_count), u(var_count);
    for(int i=0; i<N_; ++i) {
        l.segment<3>(i*3) = u_min_;
        u.segment<3>(i*3) = u_max_;
    }

    if (!solver_initialized_) {
        solver_.data()->setNumberOfVariables(var_count);
        solver_.data()->setNumberOfConstraints(var_count);
        solver_.data()->setHessianMatrix(H);
        solver_.data()->setGradient(g);
        solver_.data()->setLinearConstraintsMatrix(A);
        solver_.data()->setLowerBound(l);
        solver_.data()->setUpperBound(u);
        if(!solver_.initSolver()) return false;
        solver_initialized_ = true;
    } else {
        solver_.updateHessianMatrix(H);
        solver_.updateGradient(g);
        solver_.updateLinearConstraintsMatrix(A);
        solver_.updateBounds(l, u);
    }

    if(solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return false;

    Eigen::VectorXd solution = solver_.getSolution();
    optimal_control = ref_control_[0] + solution.head<3>();
    optimal_control = saturateControl(optimal_control);
    
    // 【新增】更新预测轨迹
    last_pred_traj_.clear();
    Eigen::VectorXd x_pred = current_state_;
    for (int i = 0; i < N_; ++i) {
        Eigen::MatrixXd A_aug, B_aug;
        buildAugmentedAB(ref_control_[i], x_pred(2), A_aug, B_aug);
        
        ControlVector u_pred = (i == 0) ? optimal_control : ref_control_[i];
        x_pred = A_aug * x_pred + B_aug * u_pred;
        
        OutputVector output;
        output << x_pred(0), x_pred(1), x_pred(2);
        last_pred_traj_.push_back(output);
    }
    
    return true;
}

bool MPCTracker::solve(ControlVector& optimal_control) {
    return solveQP(optimal_control);
}

// ✅ 在 calculateMPC 函数之前添加这些函数实现

bool MPCTracker::convertAutowareTrajectoryToInternal(
    const autoware_planning_msgs::msg::Trajectory& autoware_traj,
    std::vector<Eigen::Matrix<double, 6, 1>>& internal_traj)
{
    internal_traj.clear();
    
    if (autoware_traj.points.empty()) {
        return false;
    }
    
    for (const auto& point : autoware_traj.points) {
        Eigen::Matrix<double, 6, 1> state = Eigen::Matrix<double, 6, 1>::Zero();
        
        // 位置
        state[0] = point.pose.position.x;
        state[1] = point.pose.position.y;
        
        // 姿态 (yaw)
        tf2::Quaternion q;
        tf2::fromMsg(point.pose.orientation, q);
        double roll, pitch, yaw;
        tf2::getEulerYPR(q, yaw, pitch, roll);  // 注意顺序：yaw, pitch, roll
        state[2] = yaw;
        
        // 速度（假设只有纵向速度）
        state[3] = point.longitudinal_velocity_mps * std::cos(yaw);  // vx
        state[4] = point.longitudinal_velocity_mps * std::sin(yaw);  // vy
        
        // 角速度（默认为0）
        state[5] = 0.0;
        
        internal_traj.push_back(state);
    }
    
    return !internal_traj.empty();
}

bool MPCTracker::setReferenceTrajectory(const std::vector<Eigen::Matrix<double, 6, 1>>& ref_traj) {
    if (ref_traj.empty()) {
        return false;
    }
    
    ref_traj_.clear();
    
    // 复制轨迹，确保至少有N个点
    size_t copy_size = std::min((size_t)N_, ref_traj.size());
    for (size_t i = 0; i < copy_size; ++i) {
        ref_traj_.push_back(ref_traj[i]);
    }
    
    // 如果轨迹点数少于预测时域，用最后一个点填充
    if (ref_traj_.size() < (size_t)N_) {
        auto last_point = ref_traj_.back();
        while (ref_traj_.size() < (size_t)N_) {
            ref_traj_.push_back(last_point);
        }
    }
    
    return true;
}

bool MPCTracker::setReferenceControl(const std::vector<ControlVector>& ref_control) {
    if (ref_control.empty()) {
        return false;
    }
    
    ref_control_.clear();
    
    // 复制控制序列，确保至少有N个点
    size_t copy_size = std::min((size_t)N_, ref_control.size());
    for (size_t i = 0; i < copy_size; ++i) {
        ref_control_.push_back(ref_control[i]);
    }
    
    // 如果控制序列点数少于预测时域，用零控制填充
    if (ref_control_.size() < (size_t)N_) {
        ControlVector zero_control = ControlVector::Zero();
        while (ref_control_.size() < (size_t)N_) {
            ref_control_.push_back(zero_control);
        }
    }
    
    return true;
}



bool MPCTracker::calculateMPC(
    const autoware_control_msgs::msg::Control& current_control,
    const nav_msgs::msg::Odometry& current_kinematics,
    autoware_control_msgs::msg::Control& output,
    autoware_planning_msgs::msg::Trajectory& predicted_traj,
    std_msgs::msg::Float32MultiArray& debug_values)
{
    (void)current_control;
    
    // 更新当前状态
    current_state_[0] = current_kinematics.pose.pose.position.x;
    current_state_[1] = current_kinematics.pose.pose.position.y;
     
    tf2::Quaternion q;
    tf2::fromMsg(current_kinematics.pose.pose.orientation, q);
    double roll, pitch, yaw;
    tf2::getEulerYPR(q, roll, pitch, yaw);
    current_state_[2] = yaw;
    current_state_[3] = current_kinematics.twist.twist.linear.x;
    current_state_[4] = current_kinematics.twist.twist.linear.y;
    current_state_[5] = current_kinematics.twist.twist.angular.z;

    if (ref_traj_.empty() || ref_control_.empty()) {
        return false;
    }
    
    ControlVector optimal_control;
    if (!solve(optimal_control)) return false;

    // 填充 Control 消息
    output.lateral.steering_tire_angle = optimal_control[0];  // vx
    output.lateral.steering_tire_rotation_rate = 0.0;
    output.longitudinal.velocity = std::sqrt(std::pow(optimal_control[0], 2) + std::pow(optimal_control[1], 2));
    output.longitudinal.acceleration = optimal_control[2];  // omega
    output.longitudinal.jerk = 0.0;
    
    // 填充预测轨迹
    predicted_traj.points.clear();
    predicted_traj.header.stamp = rclcpp::Clock().now();
    predicted_traj.header.frame_id = "map";
    
    for (const auto& s : last_pred_traj_) {
        autoware_planning_msgs::msg::TrajectoryPoint p;
        p.pose.position.x = s[0];
        p.pose.position.y = s[1];
        tf2::Quaternion q_yaw;
        q_yaw.setRPY(0, 0, s[2]);
        p.pose.orientation = tf2::toMsg(q_yaw);
        p.longitudinal_velocity_mps = std::sqrt(std::pow(optimal_control[0], 2) + std::pow(optimal_control[1], 2));
        predicted_traj.points.push_back(p);
    }
    
    // 填充调试数据
    debug_values.data = {
        (float)current_state_[3], (float)current_state_[4], (float)current_state_[5],
        (float)optimal_control[0], (float)optimal_control[1], (float)optimal_control[2]
    };
    
    return true;
}
} // namespace