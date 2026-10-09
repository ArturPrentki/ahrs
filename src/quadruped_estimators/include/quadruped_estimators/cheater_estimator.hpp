#include "controller_interface/chainable_controller_interface.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "realtime_tools/realtime_buffer.hpp"

namespace quadruped_estimators {

class CheaterEstimator
    : public controller_interface::ChainableControllerInterface {
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::InterfaceConfiguration
  
  command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  // This is where we expose variables to the WBC
  std::vector<hardware_interface::StateInterface>
  on_export_state_interfaces() override;

  bool on_set_chained_mode(bool chained_mode) override;
  controller_interface::return_type
  update_reference_from_subscribers(const rclcpp::Time &time,
                                    const rclcpp::Duration &period) override;
  controller_interface::return_type
  update_and_write_commands(const rclcpp::Time &time,
                            const rclcpp::Duration &period) override;

private:
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  realtime_tools::RealtimeBuffer<nav_msgs::msg::Odometry> odom_buffer_;

  // We will export these 13 values to the Resource Manager
  // [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
  std::vector<double> base_state_;
};

} // namespace quadruped_estimators