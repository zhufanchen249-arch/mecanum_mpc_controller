cmake_minimum_required(VERSION 3.14)
project(mecanum_mpc_controller)

# 新增：指定C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(autoware_cmake REQUIRED)
autoware_package()

# 创建库
set(MECANUM_MPC_LIB ${PROJECT_NAME}_lib)
ament_auto_add_library(${MECANUM_MPC_LIB} SHARED
  src/mpc.cpp
)

# 【修正】告诉编译器头文件在include目录
target_include_directories(${MECANUM_MPC_LIB} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


# 链接外部库
target_link_libraries(${MECANUM_MPC_LIB}
  OsqpEigen::OsqpEigen
  yaml-cpp
)

# 创建可执行文件
ament_auto_add_executable(mpc_node
  src/mcp_code.cpp
)

# 链接库到可执行文件
target_link_libraries(mpc_node
  ${MECANUM_MPC_LIB}
)

# 安装
install(
  DIRECTORY param/
  DESTINATION share/${PROJECT_NAME}/param
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE
  param
)