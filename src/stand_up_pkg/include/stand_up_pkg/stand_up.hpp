#pragma once
#include "std_msgs/msg/string.hpp"
#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>

#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/time.hpp>

namespace stand_up_pkg {

class StandUp : public controller_interface::ControllerInterface {
public:
  // StandUp() =default;
  // ~StandUp() override = default;
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

private:
  Eigen::VectorXd q_;
  Eigen::VectorXd q_dot_;
  Eigen::VectorXd q_targets_[2]; // param
  
  Eigen::VectorXd tau_cmd_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> pin_joint_names_;

  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;

  double kp_;                // param
  double kd_;                // param
  double duration_;          // param

  bool done_;
  Eigen::VectorXd q_init_;
  rclcpp::Time start_time_;

  std::vector<double> reference_data_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr urdf_sub_;

  std::unordered_map<std::string, hardware_interface::LoanedCommandInterface *>
      cmd_map_;
  std::unordered_map<std::string, hardware_interface::LoanedStateInterface *>
      state_map_;

  bool urdf_received_ = false;
  static Eigen::VectorXd smooth_interpolation(const Eigen::VectorXd &q_start,
                                              const Eigen::VectorXd &q_end,
                                              double ratio);
  void update_joint_states();
  void send_command_to_hardware();
};

} // namespace stand_up_pkg
