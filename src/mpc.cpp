#include <OsqpEigen/OsqpEigen.h>
#include "mpc.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm> 
#include <tf2/LinearMath/Quaternion.h>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <autoware_auto_control_msgs/msg/ackermann_lateral_command.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace autoware::mpc_lateral_controller {

// 类型定义（需确保mpc.hpp中包含对应声明）
using StateVector = Eigen::VectorXd;
using ControlVector = Eigen::Vector3d;
using OutputVector = Eigen::Vector3d;

// 构造函数
MPCTracker::MPCTracker(const std::string& configFile, double wheelbase)
    : N_(20), dt_(0.05), wheelbase_(wheelbase),
      wheel_radius_(0.05), track_width_(0.5),
      u_min_(ControlVector::Zero()), u_max_(ControlVector::Zero()),
      x_min_(StateVector::Zero(6)), x_max_(StateVector::Zero(6)),
      current_state_(StateVector::Zero(6)),
      C_tilde_(Eigen::MatrixXd::Identity(3, 6)),
      solver_initialized_(false), last_pred_traj_() {
    
    if (!loadConfig(configFile)) {
        std::cerr << "[MPCTracker] 使用默认配置" << std::endl;
        initializeWeights();
        // 默认约束
        x_min_ << -10.0, -5.0, -M_PI, -2.0, -2.0, -2.0;
        x_max_ << 10.0, 5.0, M_PI, 2.0, 2.0, 2.0;
        u_min_ << -1.0, -1.0, -1.5;
        u_max_ << 1.0, 1.0, 1.5;
    }
    initializeMecanumModel();
    
    // 求解器配置
    solver_.settings()->setVerbosity(false);
    solver_.settings()->setWarmStart(true);
    solver_.settings()->setMaximumIterations(1000);
}

// 初始化权重矩阵
void MPCTracker::initializeWeights() {
    Q_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
    R_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
    for (int i = 0; i < N_; ++i) {
        Q_.block<3, 3>(i * 3, i * 3) = 10.0 * Eigen::MatrixXd::Identity(3, 3);
        R_.block<3, 3>(i * 3, i * 3) = 0.1 * Eigen::MatrixXd::Identity(3, 3);
    }
}

// 加载配置文件
bool MPCTracker::loadConfig(const std::string& configFile) {
    try {
        YAML::Node config = YAML::LoadFile(configFile);
        
        // 加载预测步长和采样时间
        if (config["prediction_horizon"]) {
            N_ = config["prediction_horizon"].as<int>();
            if (N_ <= 0) {
                std::cerr << "[MPCTracker] 预测步长必须为正数，使用默认值10" << std::endl;
                N_ = 10;
            }
        }
        
        if (config["sampling_time"]) {
            dt_ = config["sampling_time"].as<double>();
            if (dt_ <= 0) {
                std::cerr << "[MPCTracker] 采样时间必须为正数，使用默认值0.1" << std::endl;
                dt_ = 0.1;
            }
        }

        // 加载状态约束
        if (config["state_constraints"]) {
            auto& sc = config["state_constraints"];
            if (sc["x_min"] && sc["x_max"]) {
                std::vector<double> x_min = sc["x_min"].as<std::vector<double>>();
                std::vector<double> x_max = sc["x_max"].as<std::vector<double>>();
                
                if (x_min.size() == 6 && x_max.size() == 6) {
                    for (int i = 0; i < 6; ++i) {
                        x_min_(i) = x_min[i];
                        x_max_(i) = x_max[i];
                    }
                } else {
                    std::cerr << "[MPCTracker] 状态约束必须为6维向量，使用默认值" << std::endl;
                }
            }
        }

        // 加载控制约束
        if (config["control_constraints"]) {
            auto& cc = config["control_constraints"];
            if (cc["u_min"] && cc["u_max"]) {
                std::vector<double> u_min = cc["u_min"].as<std::vector<double>>();
                std::vector<double> u_max = cc["u_max"].as<std::vector<double>>();
                
                if (u_min.size() == 3 && u_max.size() == 3) {
                    for (int i = 0; i < 3; ++i) {
                        u_min_(i) = u_min[i];
                        u_max_(i) = u_max[i];
                    }
                } else {
                    std::cerr << "[MPCTracker] 控制约束必须为3维向量，使用默认值" << std::endl;
                }
            }
        }

        // 加载权重参数
        if (config["weights"]) {
            auto& w = config["weights"];
            double q = w["Q"].as<double>(10.0);
            double r = w["R"].as<double>(0.1);
            
            // 初始化权重矩阵
            Q_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
            R_ = Eigen::MatrixXd::Zero(3 * N_, 3 * N_);
            for (int i = 0; i < N_; ++i) {
                Q_.block<3, 3>(i * 3, i * 3) = q * Eigen::MatrixXd::Identity(3, 3);
                R_.block<3, 3>(i * 3, i * 3) = r * Eigen::MatrixXd::Identity(3, 3);
            }
        } else {
            initializeWeights();
        }
        
        // 加载麦克纳姆轮参数
        if (config["mecanum_wheel"]) {
            auto& mw = config["mecanum_wheel"];
            wheel_radius_ = mw["radius"].as<double>(wheel_radius_);
            track_width_ = mw["track_width"].as<double>(track_width_);
            if (mw["wheelbase"]) {
                wheelbase_ = mw["wheelbase"].as<double>(wheelbase_);
            }
            initializeMecanumModel();
        }

        solver_initialized_ = false;
        std::cout << "[MPCTracker] 配置文件加载成功: " << configFile << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MPCTracker] 加载配置文件失败: " << e.what() << std::endl;
        return false;
    }
}

// 初始化麦克纳姆轮模型
void MPCTracker::initializeMecanumModel() {
    double L = wheelbase_ / 2.0;    // 半轮距（前后到中心距离）
    double W = track_width_ / 2.0;  // 半轴距（左右到中心距离）
    double r = wheel_radius_;

    // 逆向运动学矩阵：[vx, vy, ω] → 四轮转速(rad/s)
    // 轮子编号：前左、前右、后左、后右
    M_.resize(4, 3);
    M_ << 1,  -1,  -(L + W),
          1,   1,   (L + W),
          1,   1,  -(L + W),
          1,  -1,   (L + W);
    M_ /= r;  // 除以轮径转换为转速

    // 正向运动学矩阵（伪逆）：四轮转速 → [vx, vy, ω]
    M_inv_ = M_.completeOrthogonalDecomposition().pseudoInverse();
    
    std::cout << "[MPCTracker] 麦克纳姆轮模型初始化完成" << std::endl;
}

// 计算麦克纳姆轮转速
std::vector<double> MPCTracker::computeWheelSpeeds(const ControlVector& u) {
    Eigen::Vector3d twist(u(0), u(1), u(2));  // u为[vx, vy, ω]
    Eigen::Vector4d wheel_speeds = M_ * twist;
    return {wheel_speeds(0), wheel_speeds(1), wheel_speeds(2), wheel_speeds(3)};
}

// 控制输入饱和处理
MPCTracker::ControlVector MPCTracker::saturateControl(const ControlVector& u) const {
    ControlVector out;
    out(0) = std::clamp(u(0), u_min_(0), u_max_(0));
    out(1) = std::clamp(u(1), u_min_(1), u_max_(1));
    out(2) = std::clamp(u(2), u_min_(2), u_max_(2));
    return out;
}

// 构建增广状态空间矩阵A和B
void MPCTracker::buildAugmentedAB(const ControlVector& ref_u, double theta,
                                  Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug) {
    const double T = dt_;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    
    // 状态转移矩阵A
    A_aug = Eigen::MatrixXd::Identity(6, 6);
    
    // 控制矩阵B
    B_aug = Eigen::MatrixXd::Zero(6, 3);
    B_aug(0, 0) = c * T;  // delta_vx 对x的影响
    B_aug(1, 0) = s * T;  // delta_vx 对y的影响
    B_aug(0, 1) = -s * T; // delta_vy 对x的影响
    B_aug(1, 1) = c * T;  // delta_vy 对y的影响
    B_aug(2, 2) = T;      // delta_omega 对theta的影响
    B_aug(3, 0) = 1;      // vx += delta_vx
    B_aug(4, 1) = 1;      // vy += delta_vy
    B_aug(5, 2) = 1;      // w += delta_w
}

// 构建Phi矩阵
Eigen::MatrixXd MPCTracker::buildPhiAug() {
    const int n = 6;
    const int p = 3;
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(p * N_, n);
    Eigen::MatrixXd A_aug, B_aug;
    Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);

    Eigen::MatrixXd C_tilde = Eigen::MatrixXd::Zero(p, n);
    C_tilde.block(0, 0, p, p) = Eigen::MatrixXd::Identity(p, p);

    for (int i = 0; i < N_; ++i) {
        const double theta_i = ref_traj_[i](2);
        const ControlVector ref_u_i = ref_control_[i];
        
        buildAugmentedAB(ref_u_i, theta_i, A_aug, B_aug);
        if (i > 0) {
            A_power = A_aug * A_power;
        }
        Phi.block(i * p, 0, p, n) = C_tilde * A_power;
    }
    return Phi;
}

// 构建Theta矩阵
Eigen::MatrixXd MPCTracker::buildThetaAug() {
    const int n = 6;
    const int m = 3;
    const int p = 3;
    Eigen::MatrixXd Theta = Eigen::MatrixXd::Zero(p * N_, m * N_);
    std::vector<Eigen::MatrixXd> A_list(N_), B_list(N_);

    for (int i = 0; i < N_; ++i) {
        Eigen::MatrixXd A_aug, B_aug;
        const double theta_i = ref_traj_[i](2);
        const ControlVector ref_u_i = ref_control_[i];
        buildAugmentedAB(ref_u_i, theta_i, A_aug, B_aug);
        A_list[i] = A_aug;
        B_list[i] = B_aug;
    }

    Eigen::MatrixXd C_tilde = Eigen::MatrixXd::Zero(p, n);
    C_tilde.block(0, 0, p, p) = Eigen::MatrixXd::Identity(p, p);

    for (int i = 0; i < N_; ++i) {
        Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);
        for (int j = 0; j <= i; ++j) {
            int idx = i - 1 - j;
            if (j > 0) {
                A_power = A_list[idx + 1] * A_power;
            }
            Theta.block(i * p, j * m, p, m) = C_tilde * A_power * B_list[j];
        }
    }
    return Theta;
}

// 求解QP问题
bool MPCTracker::solveQP(ControlVector& optimal_control) {
    if (ref_traj_.empty() || ref_control_.empty()) {
        std::cerr << "[MPCTracker] 参考轨迹或控制输入为空！" << std::endl;
        optimal_control = ControlVector::Zero();
        return false;
    }

    const int state_dim = 6;
    const int output_dim = 3;
    const int control_dim = 3;
    const int var_count = control_dim * N_;

    if (ref_traj_.size() < N_ || ref_control_.size() < N_) {
        std::cerr << "[MPCTracker] 参考轨迹或控制长度不足！" << std::endl;
        optimal_control = ControlVector::Zero();
        return false;
    }

    StateVector x_ref0 = ref_traj_[0];
    ControlVector u_ref0 = ref_control_[0];
    StateVector x_err = current_state_ - x_ref0;
    Eigen::VectorXd xi0 = x_err;

    Eigen::VectorXd Y_ref(output_dim * N_);
    for (int i = 0; i < N_; ++i) {
        Y_ref.segment<output_dim>(i * output_dim) = ref_traj_[i].head<3>();
    }

    Eigen::MatrixXd Phi = buildPhiAug();
    Eigen::MatrixXd Theta = buildThetaAug();

    if (Phi.rows() != output_dim * N_ || Phi.cols() != state_dim ||
        Theta.rows() != output_dim * N_ || Theta.cols() != var_count) {
        std::cerr << "[MPCTracker] Phi 或 Theta 矩阵维度错误！" << std::endl;
        optimal_control = ControlVector::Zero();
        return false;
    }

    // 构建QP问题的Hessian矩阵和梯度
    Eigen::MatrixXd H = Theta.transpose() * Q_ * Theta + R_;
    H += Eigen::MatrixXd::Identity(H.rows(), H.cols()) * 1e-6; // 增加微小值保证正定
    Eigen::VectorXd g = Theta.transpose() * Q_ * (Phi * xi0 - Y_ref);

    // 设置变量边界（控制输入约束）
    Eigen::VectorXd var_lower(var_count);
    Eigen::VectorXd var_upper(var_count);
    for (int i = 0; i < N_; ++i) {
        var_lower.segment<control_dim>(i * control_dim) = u_min_;
        var_upper.segment<control_dim>(i * control_dim) = u_max_;
    }

    // 构建状态约束
    const int state_constraint_count = 2 * output_dim * N_;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(state_constraint_count * control_dim);

    Eigen::VectorXd b_ineq(state_constraint_count);
    Eigen::VectorXd Phi_x0 = Phi * xi0;
    
    last_pred_traj_.clear();
    for (int i = 0; i < N_; ++i) {
        last_pred_traj_.push_back(Phi_x0.segment<3>(i * 3) + x_ref0.head<3>());
    }

    // 填充约束矩阵和边界
    for (int i = 0; i < N_; ++i) {
        const int row_start = i * 2 * output_dim;
        const Eigen::MatrixXd Theta_i = Theta.block(i * output_dim, 0, output_dim, var_count);
        const Eigen::Vector3d x_pred_bias = Phi_x0.segment<output_dim>(i * output_dim);

        // 上界约束
        for (int col = 0; col < var_count; ++col) {
            for (int row = 0; row < output_dim; ++row) {
                if (Theta_i(row, col) != 0.0) {
                    triplets.emplace_back(row_start + row, col, Theta_i(row, col));
                }
            }
        }
        b_ineq.segment<output_dim>(row_start) = x_max_.head<3>() - x_pred_bias;

        // 下界约束
        for (int col = 0; col < var_count; ++col) {
            for (int row = 0; row < output_dim; ++row) {
                if (Theta_i(row, col) != 0.0) {
                    triplets.emplace_back(row_start + output_dim + row, col, -Theta_i(row, col));
                }
            }
        }
        b_ineq.segment<output_dim>(row_start + output_dim) = -x_min_.head<3>() + x_pred_bias;
    }

    Eigen::SparseMatrix<double> A_ineq(state_constraint_count, var_count);
    A_ineq.setFromTriplets(triplets.begin(), triplets.end());

    // 初始化或更新求解器
    if (!solver_initialized_) {
        if (!solver_.data()->setNumberOfVariables(var_count)) return false;
        if (!solver_.data()->setNumberOfConstraints(state_constraint_count)) return false;
        if (!solver_.data()->setHessianMatrix(H.sparseView())) return false;
        if (!solver_.data()->setGradient(g)) return false;
        if (!solver_.data()->setLinearConstraintsMatrix(A_ineq)) return false;
        if (!solver_.data()->setLowerBound(Eigen::VectorXd::Constant(state_constraint_count, -OsqpEigen::INFTY))) return false;
        if (!solver_.data()->setUpperBound(b_ineq)) return false;
        if (!solver_.data()->setVariableLowerBound(var_lower)) return false;
        if (!solver_.data()->setVariableUpperBound(var_upper)) return false;
        if (!solver_.initSolver()) return false;
        solver_initialized_ = true;
    } else {
        // 更新求解器参数
        if (!solver_.updateHessianMatrix(H.sparseView())) {
            std::cerr << "[MPCTracker] 更新Hessian矩阵失败！" << std::endl;
            return false;
        }
        if (!solver_.updateGradient(g)) {
            std::cerr << "[MPCTracker] 更新梯度失败！" << std::endl;
            return false;
        }
        if (!solver_.updateLinearConstraintsMatrix(A_ineq)) {
            std::cerr << "[MPCTracker] 更新线性约束矩阵失败！" << std::endl;
            return false;
        }
        if (!solver_.updateBounds(Eigen::VectorXd::Constant(state_constraint_count, -OsqpEigen::INFTY), b_ineq)) {
            std::cerr << "[MPCTracker] 更新约束边界失败！" << std::endl;
            return false;
        }
        if (!solver_.updateVariableBounds(var_lower, var_upper)) {
            std::cerr << "[MPCTracker] 更新变量边界失败！" << std::endl;
            return false;
        }
    }

    // 求解QP问题
    if (!solver_.solveProblem()) {
        std::cerr << "[MPCTracker] OSQP求解失败！可能存在约束冲突。" << std::endl;
        optimal_control = ControlVector::Zero();
        return false;
    }

    // 获取最优解并计算最终控制输入
    Eigen::VectorXd solution = solver_.getSolution();
    ControlVector delta_u = solution.head<control_dim>();
    optimal_control = ref_control_[0] + delta_u;
    optimal_control = saturateControl(optimal_control);

    return true;
}

// 求解MPC主函数
bool MPCTracker::solve(ControlVector& optimal_control) {
    bool success = solveQP(optimal_control);
    if (success) {
        // 计算麦克纳姆轮转速（可根据实际需求下发）
        std::vector<double> wheel_speeds = computeWheelSpeeds(optimal_control);
        std::cout << "轮速指令: " 
                  << wheel_speeds[0] << ", " 
                  << wheel_speeds[1] << ", " 
                  << wheel_speeds[2] << ", " 
                  << wheel_speeds[3] << std::endl;
    }
    return success;
}

// 转换Autoware轨迹到内部格式
bool MPCTracker::convertAutowareTrajectoryToInternal(
    const autoware_auto_planning_msgs::msg::Trajectory& autoware_traj,
    std::vector<StateVector>& internal_traj)
{
    internal_traj.clear();
    internal_traj.reserve(autoware_traj.points.size());

    for (const auto& point : autoware_traj.points) {
        StateVector state = StateVector::Zero(6);
        
        // 位置信息
        state[0] = point.pose.position.x;  // X坐标
        state[1] = point.pose.position.y;  // Y坐标
        
        // 从四元数获取偏航角
        double yaw = tf2::getYaw(point.pose.orientation);
        state[2] = yaw;  // 偏航角
        
        // 速度信息
        state[3] = point.longitudinal_velocity_mps * cos(yaw);  // X方向速度
        state[4] = point.longitudinal_velocity_mps * sin(yaw);  // Y方向速度
        state[5] = 0.0;  // 角速度，暂时设为0
        
        internal_traj.push_back(state);
    }
    
    // 估算角速度
    if (internal_traj.size() > 1) {
        for (size_t i = 1; i < internal_traj.size() - 1; ++i) {
            double dyaw = internal_traj[i+1][2] - internal_traj[i-1][2];
            internal_traj[i][5] = dyaw / (2 * dt_);
        }
        if (!internal_traj.empty()) {
            if (internal_traj.size() > 1) {
                internal_traj[0][5] = internal_traj[1][5];
            }
            internal_traj.back()[5] = internal_traj[internal_traj.size()-2][5];
        }
    }

    return !internal_traj.empty();
}

// 设置参考轨迹
bool MPCTracker::setReferenceTrajectory(const std::vector<StateVector>& ref_traj) {
    for (const auto& point : ref_traj) {
        if (point.size() != 6) {
            std::cerr << "[MPCTracker] 参考轨迹点必须为6维向量" << std::endl;
            return false;
        }
    }
    
    if (ref_traj.size() != N_) {
        std::cerr << "[MPCTracker] 参考轨迹长度与预测步长不匹配！" << std::endl;
        return false;  
    }
    ref_traj_ = ref_traj;
    return true;
}

// 设置参考控制
bool MPCTracker::setReferenceControl(const std::vector<ControlVector>& ref_control) {
    if (ref_control.size() != N_) {
        std::cerr << "[MPCTracker] 参考控制长度与预测步长不匹配！"  << std::endl;
        return false;  
    }
    ref_control_ = ref_control;
    return true; 
}

// 核心MPC计算函数（修改debug_values参数类型）
bool MPCTracker::calculateMPC(
    const autoware_auto_control_msgs::msg::AckermannLateralCommand & current_steer,
    const nav_msgs::msg::Odometry & current_kinematics,
    autoware_auto_control_msgs::msg::AckermannLateralCommand & output,
    autoware_auto_planning_msgs::msg::Trajectory & predicted_traj,
    std_msgs::msg::Float32MultiArray & debug_values)
{
    // 1. 更新当前状态
    current_state_[0] = current_kinematics.pose.pose.position.x;
    current_state_[1] = current_kinematics.pose.pose.position.y;
    current_state_[2] = tf2::getYaw(current_kinematics.pose.pose.orientation);
    current_state_[3] = current_kinematics.twist.twist.linear.x;
    current_state_[4] = current_kinematics.twist.twist.linear.y;
    current_state_[5] = current_kinematics.twist.twist.angular.z;

    // 检查参考轨迹是否为空
    if (ref_traj_.empty()) {
        std::cerr << "[MPCTracker] 参考轨迹为空！" << std::endl;
        return false;
    }

    // 2. 寻找最近的参考轨迹点
    int closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < ref_traj_.size(); ++i) {
        double dist = std::sqrt(
            std::pow(current_state_[0] - ref_traj_[i][0], 2) +
            std::pow(current_state_[1] - ref_traj_[i][1], 2)
        );
        
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    // 距离过远则返回失败
    if (min_dist > 2.0) {
        std::cerr << "[MPCTracker] 距离参考轨迹太远: " << min_dist << " m" << std::endl;
        return false;
    }

    // 3. 截取局部参考轨迹
    std::vector<StateVector> local_ref_traj;
    int max_idx = std::min(closest_idx + N_, static_cast<int>(ref_traj_.size()));
    
    for (int i = closest_idx; i < max_idx; ++i) {
        local_ref_traj.push_back(ref_traj_[i]);
    }
    
    // 填充不足的轨迹点
    while (local_ref_traj.size() < static_cast<size_t>(N_)) {
        local_ref_traj.push_back(ref_traj_.back());
    }
    
    // 更新局部参考轨迹
    setReferenceTrajectory(local_ref_traj);

    // 4. 求解MPC
    ControlVector optimal_control;
    bool success = solve(optimal_control);
    
    if (!success) {
        std::cerr << "[MPCTracker] MPC求解失败！" << std::endl;
        return false;
    }

    // 5. 更新状态（预测下一时刻状态）
    current_state_[3] += optimal_control[0] * dt_;
    current_state_[4] += optimal_control[1] * dt_;
    current_state_[5] += optimal_control[2] * dt_;

    // 6. 填充输出控制指令
    output.lateral.steering_tire_angle = 0.0;  // 麦克纳姆轮无转向角
    output.longitudinal.velocity = sqrt(pow(current_state_[3], 2) + pow(current_state_[4], 2));

    // 7. 填充预测轨迹（用于可视化）
    predicted_traj.points.clear();
    predicted_traj.points.reserve(last_pred_traj_.size());
    
    for (const auto& state : last_pred_traj_) {
        autoware_auto_planning_msgs::msg::TrajectoryPoint point;
        point.pose.position.x = state[0];
        point.pose.position.y = state[1];
        point.pose.position.z = 0.0;
        
        // 设置偏航角
        tf2::Quaternion quat;
        quat.setRPY(0.0, 0.0, state[2]);
        point.pose.orientation.x = quat.x();
        point.pose.orientation.y = quat.y();
        point.pose.orientation.z = quat.z();
        point.pose.orientation.w = quat.w();
        
        point.longitudinal_velocity_mps = 0.0;
        predicted_traj.points.push_back(point);
    }

    // 8. 填充调试数据（适配Float32MultiArray）
    debug_values.data.clear();
    debug_values.data.push_back(current_state_[3]);  // 当前vx
    debug_values.data.push_back(current_state_[4]);  // 当前vy
    debug_values.data.push_back(current_state_[5]);  // 当前omega
    debug_values.data.push_back(optimal_control[0]); // 指令dvx
    debug_values.data.push_back(optimal_control[1]); // 指令dvy
    debug_values.data.push_back(optimal_control[2]); // 指令domega

    return true;
}

} // namespace autoware::mpc_lateral_controller