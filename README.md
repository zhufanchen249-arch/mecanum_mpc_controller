这是一个基于 **ROS 2** 和 **Autoware** 框架实现的**全向麦克纳姆轮 MPC（模型预测控制）控制器**。
（包括核心算法 `mpc.cpp`、节点实现 `mpc_code.cpp` 以及配置文件 `mecanum_car.yaml`）.

---

# Mecanum MPC Controller (ROS 2)

这是一个专为**全向麦克纳姆轮机器人**设计的模型预测控制（MPC）跟踪器。该项目深度集成于 **Autoware** 自动驾驶框架，通过优化求解器在满足物理约束的前提下，实现高精度的轨迹跟踪。

## 核心功能

* **全向运动支持**：不同于传统的阿克曼或差速模型，本控制器支持  三自由度同步控制。
* **MPC 优化算法**：基于 `OsqpEigen` 求解器，在每个控制周期内解决二次规划（QP）问题，获取最优控制增量。
* **硬件约束处理**：在配置文件中可直接设置状态量（位置、速度）和控制量（速度增量、轮速）的物理上下限。
* **直接轮速输出**：内置麦克纳姆轮逆运动学模型，可直接发布四个轮子的转速指令（rad/s）。
* **Autoware 集成**：完美适配 Autoware 的 `Trajectory` 格式和里程计接口，可作为其控制模块的替代或扩展。

## 项目结构

```text
mecanum_mpc_controller/
├── CMakeLists.txt          # 构建脚本
├── package.xml             # 依赖管理（包含 Eigen, OSQP, Autoware Msgs 等）
├── include/
│   └── mpc.hpp             # MPC 核心类声明
├── src/
│   ├── mpc.cpp             # MPC 算法逻辑与矩阵构建
│   └── mcp_code.cpp        # ROS 2 节点实现与数据同步
├── param/
│   └── mecanum_car.yaml    # 机器人物理参数与 MPC 权重配置
└── launch/
    └── mpc_launch.xml      # 节点启动与话题重映射

```

## 关键参数配置 (`param/mecanum_car.yaml`)

你可以根据实际硬件调整以下核心参数：

* **prediction_horizon**: 预测步长（默认 10），影响前瞻距离与计算量。
* **weights (Q & R)**:
* `Q`: 状态误差权重，增加该值可提高跟踪紧密度。
* `R`: 控制输入权重，增加该值可使运动更平滑。


* **mecanum_wheel**:
* `radius`: 轮子半径。
* `track_width` & `wheelbase`: 轮距与轴距，直接影响运动学模型的准确性。



## 话题接口

### 订阅 (Subscribed Topics)

* `/localization/kinematic_state` (`nav_msgs/Odometry`): 机器人当前位姿与速度。
* `/planning/scenario_planning/trajectory` (`autoware_auto_planning_msgs/Trajectory`): 参考路径。

### 发布 (Published Topics)

* `/cmd_vel` (`geometry_msgs/Twist`): 计算出的全向平移与旋转速度。
* `output/mecanum_wheel_speeds` (`std_msgs/Float32MultiArray`): 四轮转速（FL, FR, RL, RR）。
* `/mpc/debug/predicted_trajectory` (`autoware_auto_planning_msgs/Trajectory`): MPC 预测的未来 10 步路径，可在 Rviz 中可视化。

## 快速上手

1. **编译项目**：
```bash
colcon build --packages-select mecanum_mpc_controller
source install/setup.bash

```


2. **启动控制器**：
```bash
ros2 launch mecanum_mpc_controller mpc_launch.xml

```



## 依赖库

* Eigen3
* [osqp-eigen](https://github.com/robotology/osqp-eigen)
* yaml-cpp
* Autoware Universe 相关消息包

