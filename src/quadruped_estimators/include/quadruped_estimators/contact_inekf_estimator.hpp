#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <rclcpp/publisher.hpp>
#include <string>
#include <vector>

#include "controller_interface/chainable_controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "wbc_control_msgs/msg/foot_contact_flags.hpp"
// #include "wbc_control_msgs/msg/.hpp"
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
// Include the InEKF library headers
#include "InEKF.h"

#include "std_msgs/msg/string.hpp"
namespace quadruped_estimators {

class ContactInEKFEstimator
    : public controller_interface::ChainableControllerInterface {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
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
  nav_msgs::msg::Odometry get_odom_msg(const Eigen::Vector3d &pos,
                                       const Eigen::Quaterniond &quat,
                                       const Eigen::Vector3d &vel,
                                       const Eigen::Vector3d &ang_vel);

  rclcpp::Subscription<wbc_control_msgs::msg::FootContactFlags>::SharedPtr
      foot_contacts_sub;

  realtime_tools::RealtimeBuffer<std::array<bool, 4>> command_contacts;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr urdf_sub_;
  // Output exported state [pos(3), quat(4), lin_vel(3), ang_vel(3)]
  std::vector<double> base_state_;

  // Internal placeholders for kinematics (You will integrate Pinocchio / KDL
  // here)
  Eigen::Matrix4d compute_foot_fk(int leg_id,
                                  const std::vector<double> &joint_positions);
  std::shared_ptr<inekf::InEKF> filter_;
  std::shared_ptr<pinocchio::Model> pin_model_;
  std::shared_ptr<pinocchio::Data> pin_data_;
  bool urdf_received_;
};

} // namespace quadruped_estimators