cmake_minimum_required(VERSION 3.8)
project(quadruped_estimators)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()



# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(wbc_control_msgs REQUIRED)

include(FetchContent)
FetchContent_Declare(
  inekf
  GIT_REPOSITORY https://github.com/RossHartley/invariant-ekf.git
  GIT_TAG master
)
FetchContent_MakeAvailable(inekf)
FetchContent_GetProperties(inekf SOURCE_DIR inekf_source_dir)
include_directories(${inekf_source_dir}/include)

add_library(quadruped_estimators SHARED 
  # src/cheater_estimator.cpp
  src/contact_inekf_estimator.cpp
  ${inekf_source_dir}/src/InEKF.cpp
  ${inekf_source_dir}/src/RobotState.cpp
  ${inekf_source_dir}/src/NoiseParams.cpp
  ${inekf_source_dir}/src/LieGroup.cpp
)
target_include_directories(quadruped_estimators PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${inekf_source_dir}/include
)

target_compile_features(quadruped_estimators PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(
  quadruped_estimators
  rclcpp
  rclcpp_lifecycle
  nav_msgs
  controller_interface
  hardware_interface
  pluginlib
  realtime_tools
  "wbc_control_msgs"
)
target_link_libraries(quadruped_estimators
  Eigen3::Eigen
  pinocchio::pinocchio
)
# Export the plugin XML to the ament index
pluginlib_export_plugin_description_file(controller_interface plugin.xml)

install(FILES
  plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS quadruped_estimators
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()