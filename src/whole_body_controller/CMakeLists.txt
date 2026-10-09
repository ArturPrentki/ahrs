cmake_minimum_required(VERSION 3.8)
project(whole_body_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)

set(BUILD_PYTHON_INTERFACE OFF CACHE BOOL "Disable Python bindings" FORCE)
set(BUILD_WITH_URDF_SUPPORT ON CACHE BOOL "Enable URDF parsing" FORCE)
set(BUILD_WITH_COLLISION_SUPPORT OFF CACHE BOOL "Disable hpp-fcl" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)


FetchContent_Declare(
  osqp_eigen
  GIT_REPOSITORY https://github.com/robotology/osqp-eigen.git
  GIT_TAG        v0.11.0
  SYSTEM
)
FetchContent_MakeAvailable(osqp_eigen)


# 1. Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(urdfdom REQUIRED)
find_package(pinocchio REQUIRED)
find_package(osqp_vendor REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(wbc_control_msgs REQUIRED)

add_library(whole_body_controller SHARED 
  src/whole_body_controller.cpp
  src/dynamic_system.cpp
  src/optimizer.cpp
  src/control_interpolator.cpp
)

target_include_directories(whole_body_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${pinocchio_SOURCE_DIR}/include
  ${pinocchio_BINARY_DIR}/include
)

target_compile_features(whole_body_controller PUBLIC c_std_99 cxx_std_17)

# 6. Link ROS 2 ament dependencies
ament_target_dependencies(
  whole_body_controller
  "rclcpp"
  "rclcpp_lifecycle"
  "controller_interface"
  "hardware_interface"
  "pluginlib"
  "urdfdom"
  "wbc_control_msgs"
)

# 7. Link external non-ament libraries
# Note: FetchContent makes the 'pinocchio' target available directly
target_link_libraries(whole_body_controller
  Eigen3::Eigen
  pinocchio::pinocchio
  osqp::osqp
  OsqpEigen::OsqpEigen
  tf2_eigen::tf2_eigen
)


pluginlib_export_plugin_description_file(controller_interface plugin.xml)

install(TARGETS whole_body_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  controller_interface
  hardware_interface
  pluginlib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
