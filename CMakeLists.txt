cmake_minimum_required(VERSION 3.14)
project(mecanum_mpc_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找依赖（已移除旧依赖，仅保留实际用到的核心依赖）
find_package(autoware_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(autoware_auto_control_msgs REQUIRED)
find_package(autoware_auto_planning_msgs REQUIRED)

# 算法库核心依赖（明确查找，缺一不可）
find_package(OsqpEigen REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

autoware_package()

# 2. 构建核心算法库 (MPC核心逻辑)
set(MECANUM_MPC_LIB ${PROJECT_NAME}_lib)
ament_auto_add_library(${MECANUM_MPC_LIB} SHARED
  src/mpc.cpp
)

# 头文件路径（显式添加Eigen3头文件，避免编译找不到Eigen头）
target_include_directories(${MECANUM_MPC_LIB} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

# 链接库（**关键**：显式强链接Eigen3和OsqpEigen，解决链接缺失问题）
target_link_libraries(${MECANUM_MPC_LIB}
  OsqpEigen::OsqpEigen
  yaml-cpp
  Eigen3::Eigen
)

# 3. 构建ROS2节点（封装库为可执行节点，用于实际运行）
ament_auto_add_executable(mpc_node
  src/mpc_code.cpp
)

# 节点链接核心算法库
target_link_libraries(mpc_node
  ${MECANUM_MPC_LIB}
)

# 4. 安装配置文件和启动文件
install(
  DIRECTORY param/
  DESTINATION share/${PROJECT_NAME}/param
)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 自动打包（Autoware规范，包含所有依赖和安装项）
ament_auto_package(INSTALL_TO_SHARE param launch)