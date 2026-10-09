cmake_minimum_required(VERSION 3.14)
project(stand_up_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(BUILD_PYTHON_INTERFACE OFF CACHE BOOL "Disable Python bindings" FORCE)
set(BUILD_WITH_URDF_SUPPORT ON CACHE BOOL "Enable URDF parsing" FORCE)
set(BUILD_WITH_COLLISION_SUPPORT OFF CACHE BOOL "Disable hpp-fcl" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(urdfdom REQUIRED) # Required by Pinocchio for URDF support
find_package(pinocchio REQUIRED)

find_package(wbc_control_msgs REQUIRED)
add_library(stand_up_pkg SHARED src/down_then_stand.cpp)

target_include_directories(stand_up_pkg PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${pinocchio_SOURCE_DIR}/include
  ${pinocchio_BINARY_DIR}/include
)

target_compile_features(stand_up_pkg PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  stand_up_pkg
  "rclcpp"
  "rclcpp_lifecycle"
  "controller_interface"
  "hardware_interface"
  "pluginlib"
  "urdfdom"
  "wbc_control_msgs"
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(stand_up_pkg PRIVATE "STAND_UP_PKG_BUILDING_LIBRARY")

target_link_libraries(stand_up_pkg
  Eigen3::Eigen
  pinocchio::pinocchio
)

pluginlib_export_plugin_description_file(controller_interface plugin.xml)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS stand_up_pkg
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

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
