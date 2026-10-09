#include "quadruped_estimators/cheater_estimator.hpp"
#include <rclcpp/logging.hpp>

namespace quadruped_estimators {

controller_interface::InterfaceConfiguration
CheaterEstimator::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::InterfaceConfiguration
CheaterEstimator::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::CallbackReturn CheaterEstimator::on_init() {
  base_state_.resize(13, 0.0);
  base_state_[6] = 1.0; // quat_w default
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
CheaterEstimator::on_configure(const rclcpp_lifecycle::State &) {
  odom_sub_ = get_node()->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", rclcpp::SystemDefaultsQoS(),
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        odom_buffer_.writeFromNonRT(*msg);
      });
  return controller_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
CheaterEstimator::on_export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> interfaces;

  // clang-format off
  std::vector<std::string> names = { 
      "pos_x", "pos_y", "pos_z",
      "quat_x", "quat_y", "quat_z", "quat_w",
      "lin_vel_x", "lin_vel_y", "lin_vel_z",
      "ang_vel_x", "ang_vel_y", "ang_vel_z"
    };
  // clang-format on

  for (size_t i = 0; i < names.size(); ++i) {
    interfaces.emplace_back(hardware_interface::StateInterface(
        get_node()->get_name(), names[i], &base_state_[i]));
  }
  return interfaces;
}

bool CheaterEstimator::on_set_chained_mode(bool /*chained_mode*/) {
  return true;
}

controller_interface::return_type
CheaterEstimator::update_reference_from_subscribers(const rclcpp::Time &,
                                                    const rclcpp::Duration &) {
  auto odom_msg = odom_buffer_.readFromRT();
  if (!odom_msg)
    return controller_interface::return_type::OK;

  // Write directly into reference_interfaces_ using standard pointer access
  base_state_[0] = odom_msg->pose.pose.position.x;
  base_state_[1] = odom_msg->pose.pose.position.y;
  base_state_[2] = odom_msg->pose.pose.position.z;

  base_state_[3] = odom_msg->pose.pose.orientation.x;
  base_state_[4] = odom_msg->pose.pose.orientation.y;
  base_state_[5] = odom_msg->pose.pose.orientation.z;
  base_state_[6] = odom_msg->pose.pose.orientation.w;

  base_state_[7] = odom_msg->twist.twist.linear.x;
  base_state_[8] = odom_msg->twist.twist.linear.y;
  base_state_[9] = odom_msg->twist.twist.linear.z;

  base_state_[10] = odom_msg->twist.twist.angular.x;
  base_state_[11] = odom_msg->twist.twist.angular.y;
  base_state_[12] = odom_msg->twist.twist.angular.z;

  return controller_interface::return_type::OK;
}

controller_interface::return_type
CheaterEstimator::update_and_write_commands(const rclcpp::Time &,
                                            const rclcpp::Duration &) {
  return controller_interface::return_type::OK;
}

} // namespace quadruped_estimators

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(quadruped_estimators::CheaterEstimator,
                       controller_interface::ChainableControllerInterface)