cmake_minimum_required(VERSION 3.14)
project(mecanum_mpc_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(autoware_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)  # 因为 mpc_lib 需要
find_package(autoware_control_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)

find_package(OsqpEigen REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)

autoware_package()

# 核心算法库
ament_auto_add_library(mpc_lib SHARED
  src/mpc.cpp
)
target_include_directories(mpc_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(mpc_lib
  OsqpEigen::OsqpEigen
  yaml-cpp
  Eigen3::Eigen
)
ament_target_dependencies(mpc_lib
  "rclcpp"
  "std_msgs"
  "nav_msgs"
  "geometry_msgs"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
  "autoware_control_msgs"
  "autoware_planning_msgs"
)

# 主节点
ament_auto_add_executable(mpc_node
  src/mpc_code.cpp
)
target_link_libraries(mpc_node mpc_lib)
ament_target_dependencies(mpc_node "ament_index_cpp")

# 简单测试节点
ament_auto_add_executable(simple_test
  src/simple_test.cpp
)
target_link_libraries(simple_test mpc_lib)
ament_target_dependencies(simple_test "ament_index_cpp")

# 安装
install(DIRECTORY param/ DESTINATION share/${PROJECT_NAME}/param)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts)
ament_auto_package(INSTALL_TO_SHARE param launch scripts)