#pragma once
#include "std_msgs/msg/string.hpp"
// #include "wbc_control_msgs/msg/foot"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <controller_interface/controller_interface.hpp>

#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/time.hpp>

namespace stand_up_pkg {

class DownThenStand : public controller_interface::ControllerInterface {
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  controller_interface::return_type
  update(const rclcpp::Time &time, const rclcpp::Duration &period) override;

  using Vector12d = Eigen::Matrix<double, 12, 1>;
  using Vector18d = Eigen::Matrix<double, 18, 1>;
  using Vector19d = Eigen::Matrix<double, 19, 1>;
  
private:
  enum class StandUpPhase { Down, Stand };
  StandUpPhase current_phase;

  Vector19d q_;
  Vector18d q_dot_;

  Vector12d target_poses[2]; // param

  Vector12d ff_torque;
  Vector12d current_joint_pose;

  std::vector<std::string> joint_names_;
  std::vector<std::string> pin_joint_names_;

  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;

  double kp_;       // param
  double kd_;       // param
  double duration_; // param

  bool done_;
  Vector12d q_init_;
  rclcpp::Time start_time_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr urdf_sub_;
  bool urdf_received_ = false;

  std::unordered_map<std::string, hardware_interface::LoanedCommandInterface *>
      cmd_map_;
  std::unordered_map<std::string, hardware_interface::LoanedStateInterface *>
      state_map_;

  static Vector12d smooth_interpolation(const Vector12d &q_start,
                                        const Vector12d &q_end, double ratio);
  void update_joint_states();
  void send_command_to_hardware(const Vector12d &desired_position);
};

} // namespace stand_up_pkg
