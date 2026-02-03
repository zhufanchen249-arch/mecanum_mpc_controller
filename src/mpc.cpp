#include "mpc.hpp"
// 必要基础头文件（无冗余，均为代码实际用到）
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <tf2/utils.h>

namespace autoware::mpc_lateral_controller {

// 【核心】构造函数初始化列表 → 严格匹配mpc.hpp的成员变量顺序，彻底消除-Wreorder警告
MPCTracker::MPCTracker(const std::string& configFile, double wheelbase)
    : N_(20),
      dt_(0.05),
      wheelbase_(wheelbase),
      wheel_radius_(0.05),
      track_width_(0.5),
      u_min_(ControlVector::Zero()),
      u_max_(ControlVector::Zero()),
      x_min_(Eigen::VectorXd::Zero(6)),
      x_max_(Eigen::VectorXd::Zero(6)),
      current_state_(Eigen::VectorXd::Zero(6)),
      Q_(Eigen::MatrixXd::Zero(3 * N_, 3 * N_)),
      R_(Eigen::MatrixXd::Zero(3 * N_, 3 * N_)),
      M_(Eigen::MatrixXd::Zero(4, 3)),
      M_inv_(Eigen::MatrixXd::Zero(3, 4)),
      C_tilde_(Eigen::MatrixXd::Identity(3, 6)),
      solver_initialized_(false),
      last_pred_traj_(),
      ref_traj_(),
      ref_control_(),
      solver_()
{
    // 初始化默认参数（约束+权重）
    initializeWeights();
    x_min_ << -10.0, -5.0, -M_PI, -2.0, -2.0, -2.0;
    x_max_ << 10.0, 5.0, M_PI, 2.0, 2.0, 2.0;
    u_min_ << -1.0, -1.0, -1.5;
    u_max_ << 1.0, 1.0, 1.5;

    // 加载配置文件，失败则用默认
    if (!loadConfig(configFile)) {
        std::cerr << "[MPCTracker] 配置文件加载失败，使用默认参数" << std::endl;
    }
    initializeMecanumModel();

    // 新版OsqpEigen基础配置（无旧版API，适配最新版本）
    solver_.settings()->setVerbosity(false);
    solver_.settings()->setWarmStart(true);
    solver_.settings()->setMaximumIterations(1000); // 显式设置最大迭代数，提升求解稳定性
}

// 初始化MPC状态/控制权重矩阵Q/R（动态适配N_，非固定维度）
void MPCTracker::initializeWeights() {
    Q_.resize(3 * N_, 3 * N_);
    R_.resize(3 * N_, 3 * N_);
    Q_.setZero();
    R_.setZero();

    // 状态权重放大，控制权重缩小，平衡跟踪精度和控制平滑性
    for (int i = 0; i < N_; ++i) {
        Q_.block<3, 3>(i * 3, i * 3) = 10.0 * Eigen::MatrixXd::Identity(3, 3);
        R_.block<3, 3>(i * 3, i * 3) = 0.1 * Eigen::MatrixXd::Identity(3, 3);
    }
}

// 加载YAML配置文件 → 修复auto&绑定临时变量问题，改为值拷贝，无编译错误
bool MPCTracker::loadConfig(const std::string& configFile) {
    try {
        // 加载配置文件，捕获所有yaml解析异常
        YAML::Node config = YAML::LoadFile(configFile);

        // 1. 加载MPC核心参数，做合法性检查（避免非法值）
        if (config["prediction_horizon"]) {
            N_ = config["prediction_horizon"].as<int>();
            if (N_ <= 0) N_ = 20; // 非法值重置为默认
        }
        if (config["sampling_time"]) {
            dt_ = config["sampling_time"].as<double>();
            if (dt_ <= 0) dt_ = 0.05; // 非法值重置为默认
        }

        // 2. 加载状态约束 → 改为值拷贝，避免引用临时变量（核心修复点）
        if (config["state_constraints"]) {
            YAML::Node sc = config["state_constraints"]; // 值拷贝，非引用
            if (sc["x_min"] && sc["x_max"]) {
                std::vector<double> x_min_vec = sc["x_min"].as<std::vector<double>>();
                std::vector<double> x_max_vec = sc["x_max"].as<std::vector<double>>();
                if (x_min_vec.size() == 6 && x_max_vec.size() == 6) {
                    for (int i = 0; i < 6; ++i) {
                        x_min_(i) = x_min_vec[i];
                        x_max_(i) = x_max_vec[i];
                    }
                }
            }
        }

        // 3. 加载控制约束 → 同样值拷贝，无临时变量引用
        if (config["control_constraints"]) {
            YAML::Node cc = config["control_constraints"]; // 值拷贝，非引用
            if (cc["u_min"] && cc["u_max"]) {
                std::vector<double> u_min_vec = cc["u_min"].as<std::vector<double>>();
                std::vector<double> u_max_vec = cc["u_max"].as<std::vector<double>>();
                if (u_min_vec.size() == 3 && u_max_vec.size() == 3) {
                    for (int i = 0; i < 3; ++i) {
                        u_min_(i) = u_min_vec[i];
                        u_max_(i) = u_max_vec[i];
                    }
                }
            }
        }

        // 4. 加载权重参数
        if (config["weights"]) {
            YAML::Node w = config["weights"]; // 值拷贝，非引用
            double q = w["Q"].as<double>(10.0);
            double r = w["R"].as<double>(0.1);
            Q_.setZero(3 * N_, 3 * N_);
            R_.setZero(3 * N_, 3 * N_);
            for (int i = 0; i < N_; ++i) {
                Q_.block<3, 3>(i * 3, i * 3) = q * Eigen::MatrixXd::Identity(3, 3);
                R_.block<3, 3>(i * 3, i * 3) = r * Eigen::MatrixXd::Identity(3, 3);
            }
        } else {
            initializeWeights(); // 无权重配置则用默认
        }

        // 5. 加载麦克纳姆轮硬件参数
        if (config["mecanum_wheel"]) {
            YAML::Node mw = config["mecanum_wheel"]; // 值拷贝，非引用
            wheel_radius_ = mw["radius"].as<double>(wheel_radius_);
            track_width_ = mw["track_width"].as<double>(track_width_);
            if (mw["wheelbase"]) wheelbase_ = mw["wheelbase"].as<double>(wheelbase_);
        }

        // 配置更新后，重新初始化核心组件
        initializeWeights();
        initializeMecanumModel();
        solver_initialized_ = false; // 重置求解器，后续重新初始化
        std::cout << "[MPCTracker] 配置文件加载成功: " << configFile << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MPCTracker] 配置文件加载失败: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[MPCTracker] 配置文件加载失败：未知异常" << std::endl;
        return false;
    }
}

// 初始化麦克纳姆轮逆向/正向运动学模型
void MPCTracker::initializeMecanumModel() {
    double L = wheelbase_ / 2.0;  // 半轮距（前后中心到机器人中心）
    double W = track_width_ / 2.0;// 半轴距（左右中心到机器人中心）
    double r = wheel_radius_;     // 车轮半径

    // 逆向运动学矩阵：[vx, vy, omega] → 四轮转速(rad/s)，轮子编号：前左、前右、后左、后右
    M_.resize(4, 3);
    M_ << 1,  -1,  -(L + W),
          1,   1,   (L + W),
          1,   1,  -(L + W),
          1,  -1,   (L + W);
    M_ /= r; // 除以轮径转换为角速度

    // 正向运动学伪逆矩阵：四轮转速 → [vx, vy, omega]（应对非方阵求逆）
    M_inv_ = M_.completeOrthogonalDecomposition().pseudoInverse();
}

// 控制输入饱和处理 → 限制在硬件允许范围内，防止执行器过载
MPCTracker::ControlVector MPCTracker::saturateControl(const ControlVector& u) const {
    ControlVector out;
    out(0) = std::clamp(u(0), u_min_(0), u_max_(0)); // vx饱和
    out(1) = std::clamp(u(1), u_min_(1), u_max_(1)); // vy饱和
    out(2) = std::clamp(u(2), u_min_(2), u_max_(2)); // omega饱和
    return out;
}

// 构建MPC增广状态空间矩阵A_aug(6x6)和控制矩阵B_aug(6x3)
void MPCTracker::buildAugmentedAB(const ControlVector& ref_u, double theta,
                                  Eigen::MatrixXd& A_aug, Eigen::MatrixXd& B_aug) {
    (void)ref_u; // 预留参数，兼容后续扩展
    const double T = dt_;
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    // 状态转移矩阵A → 恒等矩阵（离散时间近似）
    A_aug = Eigen::MatrixXd::Identity(6, 6);
    // 控制矩阵B → 麦克纳姆轮全向运动特性，关联控制输入与状态变化
    B_aug = Eigen::MatrixXd::Zero(6, 3);
    B_aug(0, 0) = c * T;  B_aug(0, 1) = -s * T; // x位置由vx/vy共同影响
    B_aug(1, 0) = s * T;  B_aug(1, 1) = c * T;  // y位置由vx/vy共同影响
    B_aug(2, 2) = T;                            // 偏航角由omega影响
    B_aug(3, 0) = 1.0;                           // vx直接更新
    B_aug(4, 1) = 1.0;                           // vy直接更新
    B_aug(5, 2) = 1.0;                           // omega直接更新
}

// 构建Phi矩阵 → 状态到输出的映射矩阵 (3N x 6)
Eigen::MatrixXd MPCTracker::buildPhiAug() {
    const int n = 6;  // 状态维度
    const int p = 3;  // 输出维度（x,y,yaw）
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(p * N_, n);
    Eigen::MatrixXd A_aug, B_aug;
    Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n); // A的幂次，初始为恒等矩阵

    // 输出映射矩阵C_tilde → 只取状态的前3维（x,y,yaw）
    Eigen::MatrixXd C_tilde = Eigen::MatrixXd::Zero(p, n);
    C_tilde.block(0, 0, p, p) = Eigen::MatrixXd::Identity(p, p);

    // 遍历预测步长，构建Phi矩阵
    for (int i = 0; i < N_; ++i) {
        buildAugmentedAB(ref_control_[i], ref_traj_[i](2), A_aug, B_aug);
        if (i > 0) A_power = A_aug * A_power; // 计算A的i次幂
        Phi.block(i * p, 0, p, n) = C_tilde * A_power;
    }
    return Phi;
}

// 构建Theta矩阵 → 控制到输出的映射矩阵 (3N x 3N)
Eigen::MatrixXd MPCTracker::buildThetaAug() {
    const int n = 6;  // 状态维度
    const int m = 3;  // 控制维度
    const int p = 3;  // 输出维度
    Eigen::MatrixXd Theta = Eigen::MatrixXd::Zero(p * N_, m * N_);
    std::vector<Eigen::MatrixXd> A_list(N_), B_list(N_);

    // 预计算所有步长的A/B矩阵，避免重复计算
    for (int i = 0; i < N_; ++i) {
        Eigen::MatrixXd A_aug, B_aug;
        buildAugmentedAB(ref_control_[i], ref_traj_[i](2), A_aug, B_aug);
        A_list[i] = A_aug;
        B_list[i] = B_aug;
    }

    // 输出映射矩阵C_tilde
    Eigen::MatrixXd C_tilde = Eigen::MatrixXd::Zero(p, n);
    C_tilde.block(0, 0, p, p) = Eigen::MatrixXd::Identity(p, p);

    // 遍历预测步长，构建Theta矩阵
    for (int i = 0; i < N_; ++i) {
        Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);
        for (int j = 0; j <= i; ++j) {
            int idx = i - j;
            if (j > 0) A_power = A_list[idx] * A_power;
            Theta.block(i * p, j * m, p, m) = C_tilde * A_power * B_list[j];
        }
    }
    return Theta;
}

// 求解QP二次规划问题 → 适配新版OsqpEigen API，无旧版调用，参数检查完善
bool MPCTracker::solveQP(ControlVector& optimal_control) {
    // 前置检查：参考轨迹/控制非空，且长度匹配预测步长
    if (ref_traj_.empty() || ref_control_.empty() ||
        ref_traj_.size() != (size_t)N_ || ref_control_.size() != (size_t)N_) {
        std::cerr << "[MPCTracker] 参考轨迹/控制为空或长度不匹配！" << std::endl;
        optimal_control = ControlVector::Zero();
        return false;
    }

    const int output_dim = 3;
    const int control_dim = 3;
    const int var_count = control_dim * N_; // 优化变量总数：3N
    const int state_dim = 6;               // 状态维度

    // 计算当前状态与参考轨迹的偏差
    Eigen::VectorXd xi0 = current_state_ - ref_traj_[0];
    // 构建参考输出向量Y_ref (3N x 1) → 只取前3维（x,y,yaw）
    Eigen::VectorXd Y_ref(output_dim * N_);
    for (int i = 0; i < N_; ++i) {
        Y_ref.segment<output_dim>(i * output_dim) = ref_traj_[i].head<output_dim>();
    }

    // 构建核心映射矩阵Phi和Theta
    Eigen::MatrixXd Phi = buildPhiAug();
    Eigen::MatrixXd Theta = buildThetaAug();

    // 构建QP问题的Hessian矩阵H和梯度向量g
    Eigen::SparseMatrix<double> H = (Theta.transpose() * Q_ * Theta + R_).sparseView();
    // 增加微小正定项，避免H矩阵奇异导致求解失败
    H += Eigen::SparseMatrix<double>(var_count, var_count).setIdentity() * 1e-6;
    Eigen::VectorXd g = Theta.transpose() * Q_ * (Phi * xi0 - Y_ref);

    // 构建控制输入约束：变量上下界 (3N x 1)
    Eigen::VectorXd var_lower(var_count);
    Eigen::VectorXd var_upper(var_count);
    for (int i = 0; i < N_; ++i) {
        var_lower.segment<control_dim>(i * control_dim) = u_min_;
        var_upper.segment<control_dim>(i * control_dim) = u_max_;
    }

    // 构建状态约束：线性约束矩阵A_ineq和边界b_ineq
    const int constraint_count = output_dim * N_ * 2; // 每个输出维度有上下界
    Eigen::SparseMatrix<double> A_ineq(constraint_count, var_count);
    Eigen::VectorXd b_ineq(constraint_count);
    Eigen::VectorXd Phi_xi0 = Phi * xi0;

    // 填充约束矩阵和边界（稀疏矩阵，只存非零元素）
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(constraint_count * control_dim);
    for (int i = 0; i < N_; ++i) {
        const int row_base = i * 2 * output_dim;
        const Eigen::MatrixXd Theta_i = Theta.block(i * output_dim, 0, output_dim, var_count);
        const Eigen::VectorXd pred_bias = Phi_xi0.segment(output_dim, i * output_dim);

        // 上界约束：Theta * u ≤ x_max - pred_bias
        for (int col = 0; col < var_count; ++col) {
            for (int row = 0; row < output_dim; ++row) {
                if (std::fabs(Theta_i(row, col)) > 1e-9) {
                    triplets.emplace_back(row_base + row, col, Theta_i(row, col));
                }
            }
        }
        b_ineq.segment<output_dim>(row_base) = x_max_.head<output_dim>() - pred_bias;

        // 下界约束：-Theta * u ≤ -x_min + pred_bias → Theta * u ≥ x_min - pred_bias
        for (int col = 0; col < var_count; ++col) {
            for (int row = 0; row < output_dim; ++row) {
                if (std::fabs(Theta_i(row, col)) > 1e-9) {
                    triplets.emplace_back(row_base + output_dim + row, col, -Theta_i(row, col));
                }
            }
        }
        b_ineq.segment<output_dim>(row_base + output_dim) = -x_min_.head<output_dim>() + pred_bias;
    }
    A_ineq.setFromTriplets(triplets.begin(), triplets.end());

    // 初始化/更新OsqpEigen求解器 → 新版API，带参数合法性检查
    if (!solver_initialized_) {
        // 首次初始化：设置变量/约束数，绑定矩阵/向量
        if (!solver_.data()->setNumberOfVariables(var_count)) return false;
        if (!solver_.data()->setNumberOfConstraints(constraint_count)) return false;
        if (!solver_.data()->setHessianMatrix(H)) return false;
        if (!solver_.data()->setGradient(g)) return false;
        if (!solver_.data()->setLinearConstraintsMatrix(A_ineq)) return false;
        // 设置约束边界：线性约束下界为负无穷，上界为b_ineq
        if (!solver_.data()->setLowerBound(Eigen::VectorXd::Constant(constraint_count, -OsqpEigen::INFTY))) return false;
        if (!solver_.data()->setUpperBound(b_ineq)) return false;
        // 设置优化变量边界
        if (!solver_.data()->setVariableLowerBound(var_lower)) return false;
        if (!solver_.data()->setVariableUpperBound(var_upper)) return false;
        // 初始化求解器
        if (!solver_.initSolver()) {
            std::cerr << "[MPCTracker] OsqpEigen求解器初始化失败！" << std::endl;
            solver_initialized_ = false;
            return false;
        }
        solver_initialized_ = true;
    } else {
        // 后续更新：只更新变化的矩阵/向量，提升效率
        if (!solver_.updateHessianMatrix(H)) return false;
        if (!solver_.updateGradient(g)) return false;
        if (!solver_.updateLinearConstraintsMatrix(A_ineq)) return false;
        if (!solver_.updateBounds(Eigen::VectorXd::Constant(constraint_count, -OsqpEigen::INFTY), b_ineq)) return false;
        if (!solver_.updateVariableBounds(var_lower, var_upper)) return false;
    }

    // 求解QP问题 → 新版API，检查求解状态
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
        std::cerr << "[MPCTracker] QP问题求解失败！" << std::endl;
        optimal_control = ControlVector::Zero();
        return false;
    }

    // 获取最优解，取第一个控制步作为输出（MPC滚动优化）
    Eigen::VectorXd solution = solver_.getSolution();
    ControlVector delta_u = solution.head<control_dim>();
    optimal_control = ref_control_[0] + delta_u;
    optimal_control = saturateControl(optimal_control); // 最后一次饱和处理

    return true;
}

// MPC求解主入口 → 封装QP求解，兼容后续扩展
bool MPCTracker::solve(ControlVector& optimal_control) {
    return solveQP(optimal_control);
}

// 将Autoware标准轨迹转换为MPC内部6维状态轨迹 → 类型统一为StateVector
bool MPCTracker::convertAutowareTrajectoryToInternal(
    const autoware_auto_planning_msgs::msg::Trajectory& autoware_traj,
    std::vector<StateVector>& internal_traj)
{
    internal_traj.clear();
    internal_traj.reserve(autoware_traj.points.size());

    for (const auto& point : autoware_traj.points) {
        StateVector state = StateVector::Zero(6);
        // 位置与偏航角
        state[0] = point.pose.position.x;
        state[1] = point.pose.position.y;
        state[2] = tf2::getYaw(point.pose.orientation);
        // 速度分解：将纵向速度分解为x/y方向的速度（全向运动）
        double v = point.longitudinal_velocity_mps;
        state[3] = v * std::cos(state[2]);
        state[4] = v * std::sin(state[2]);
        state[5] = 0.0; // 角速度默认0，后续可由轨迹差分估算
        internal_traj.push_back(state);
    }

    // 差分估算角速度，提升轨迹精度
    if (internal_traj.size() > 2) {
        for (size_t i = 1; i < internal_traj.size() - 1; ++i) {
            double dyaw = internal_traj[i+1][2] - internal_traj[i-1][2];
            internal_traj[i][5] = dyaw / (2 * dt_);
        }
        internal_traj[0][5] = internal_traj[1][5];
        internal_traj.back()[5] = internal_traj[internal_traj.size()-2][5];
    }

    return !internal_traj.empty();
}

// 设置MPC参考轨迹 → 与头文件声明一致，类型为StateVector
bool MPCTracker::setReferenceTrajectory(const std::vector<StateVector>& ref_traj) {
    if (ref_traj.empty() || ref_traj.size() != (size_t)N_) {
        std::cerr << "[MPCTracker] 参考轨迹长度与预测步长不匹配！" << std::endl;
        return false;
    }
    ref_traj_ = ref_traj;
    return true;
}

// 设置MPC参考控制 → 与头文件声明一致
bool MPCTracker::setReferenceControl(const std::vector<ControlVector>& ref_control) {
    if (ref_control.empty() || ref_control.size() != (size_t)N_) {
        std::cerr << "[MPCTracker] 参考控制长度与预测步长不匹配！" << std::endl;
        return false;
    }
    ref_control_ = ref_control;
    return true;
}

// 计算麦克纳姆轮四轮转速 → 逆向运动学
std::vector<double> MPCTracker::computeWheelSpeeds(const ControlVector& u) {
    Eigen::Vector4d wheel_speeds = M_ * u;
    return {wheel_speeds(0), wheel_speeds(1), wheel_speeds(2), wheel_speeds(3)};
}

// 核心MPC计算接口 → 与头文件声明1:1匹配，无参数类型错误
bool MPCTracker::calculateMPC(
    const autoware_auto_control_msgs::msg::AckermannLateralCommand & current_steer,
    const nav_msgs::msg::Odometry & current_kinematics,
    autoware_auto_control_msgs::msg::AckermannLateralCommand & output,
    autoware_auto_planning_msgs::msg::Trajectory & predicted_traj,
    std_msgs::msg::Float32MultiArray & debug_values)
{
    (void)current_steer; // 预留参数，麦克纳姆轮无转向指令

    // 1. 从里程计更新机器人当前6维状态
    current_state_[0] = current_kinematics.pose.pose.position.x;
    current_state_[1] = current_kinematics.pose.pose.position.y;
    current_state_[2] = tf2::getYaw(current_kinematics.pose.pose.orientation);
    current_state_[3] = current_kinematics.twist.twist.linear.x;
    current_state_[4] = current_kinematics.twist.twist.linear.y;
    current_state_[5] = current_kinematics.twist.twist.angular.z;

    // 2. 前置检查：参考轨迹非空
    if (ref_traj_.empty()) {
        std::cerr << "[MPCTracker] 参考轨迹为空，无法计算MPC！" << std::endl;
        return false;
    }

    // 3. 求解MPC，得到最优控制指令[vx, vy, omega]
    ControlVector optimal_control;
    if (!solve(optimal_control)) {
        std::cerr << "[MPCTracker] MPC求解失败，返回false！" << std::endl;
        return false;
    }

    // 4. 填充输出控制指令 → 适配AckermannLateralCommand格式（麦克纳姆轮无转向）
    output.lateral.steering_tire_angle = 0.0; // 麦克纳姆轮无需转向，置0
    output.lateral.steering_tire_rotation_rate = 0.0;
    // 纵向速度为x/y方向速度的合速度
    output.longitudinal.velocity = std::sqrt(std::pow(optimal_control[0], 2) + std::pow(optimal_control[1], 2));
    output.longitudinal.acceleration = 0.0; // 预留加速度，兼容后续扩展

    // 5. 填充MPC预测轨迹 → 用于Rviz可视化
    predicted_traj.points.clear();
    predicted_traj.points.reserve(N_);
    StateVector pred_state = current_state_;
    Eigen::MatrixXd A_aug, B_aug;
    for (int i = 0; i < N_; ++i) {
        buildAugmentedAB(ref_control_[i], pred_state[2], A_aug, B_aug);
        pred_state = A_aug * pred_state + B_aug * optimal_control;

        autoware_auto_planning_msgs::msg::TrajectoryPoint p;
        p.pose.position.x = pred_state[0];
        p.pose.position.y = pred_state[1];
        p.pose.position.z = 0.0;
        // 预测轨迹偏航角转四元数
        tf2::Quaternion quat;
        quat.setRPY(0.0, 0.0, pred_state[2]);
        p.pose.orientation = tf2::toMsg(quat);
        p.longitudinal_velocity_mps = output.longitudinal.velocity;
        predicted_traj.points.push_back(p);
    }
    last_pred_traj_.clear();
    for (const auto& p : predicted_traj.points) {
        OutputVector s;
        s << p.pose.position.x, p.pose.position.y, tf2::getYaw(p.pose.orientation);
        last_pred_traj_.push_back(s);
    }

    // 6. 填充调试数据 → 适配std_msgs::Float32MultiArray，无Stamped
    debug_values.data.clear();
    debug_values.data = {
        (float)current_state_[3],  // 当前x方向速度vx
        (float)current_state_[4],  // 当前y方向速度vy
        (float)current_state_[5],  // 当前角速度omega
        (float)optimal_control[0], // MPC指令vx
        (float)optimal_control[1], // MPC指令vy
        (float)optimal_control[2]  // MPC指令omega
    };

    return true;
}

} // namespace autoware::mpc_lateral_controller