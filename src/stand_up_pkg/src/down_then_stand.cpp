#include "stand_up_pkg/down_then_stand.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <controller_interface/controller_interface_base.hpp>
#include <cstddef>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/time.hpp>
#include <tuple>
#include <vector>

namespace stand_up_pkg { // namespace stand_up_pkg

controller_interface::CallbackReturn DownThenStand::on_init() {
  try {
    auto_declare<std::vector<std::string>>("joints",
                                           std::vector<std::string>());
    // Update default target targets to FR, FL, RR, RL order
    auto_declare<std::vector<double>>("target_stand",
                                      {-0.1, 0.8, -1.5,  // FR (HAA, HFE, KFE)
                                       0.1, 0.8, -1.5,   // FL
                                       -0.1, 1.0, -1.5,  // RR
                                       0.1, 1.0, -1.5}); // RL
    auto_declare<std::vector<double>>("target_down", {-0.01, 1.27, -2.8, ///
                                                      0.01, 1.27, -2.8,  //
                                                      -0.3, 1.31, -2.8,  //
                                                      0.3, 1.31, -2.8});

    auto_declare<double>("kp", 10.0);
    auto_declare<double>("kd", 2.0);
    auto_declare<double>("duration", 3.0);

  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Exception thrown during on_init: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DownThenStand::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  std::vector<double> target_pos1 =
      get_node()->get_parameter("target_down").as_double_array();
  std::vector<double> target_pos2 =
      get_node()->get_parameter("target_stand").as_double_array();
  kp_ = get_node()->get_parameter("kp").as_double();
  kd_ = get_node()->get_parameter("kd").as_double();
  duration_ = get_node()->get_parameter("duration").as_double();

  if (joint_names_.empty() || target_pos1.size() < 12 ||
      target_pos2.size() < 12) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Invalid 'joints' or 'target' parameters!");
    return controller_interface::CallbackReturn::ERROR;
  }

  // NOTE: We remove the raw q_target_[i] = target_pos[i] loop from here.
  // We must do it INSIDE the lambda once we know Pinocchio's URDF layout.

  auto qos = rclcpp::QoS(1).transient_local();
  // Capture target_pos into the lambda
  urdf_sub_ = get_node()->create_subscription<std_msgs::msg::String>(
      "/robot_description", qos,
      [this, target_pos1,
       target_pos2](const std_msgs::msg::String::SharedPtr msg) {
        if (!urdf_received_) {
          std::string base_name = "trunk";

          pinocchio::urdf::buildModelFromXML(
              msg->data, pinocchio::JointModelFreeFlyer(), pin_model_);

          pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
          q_.setZero(pin_model_.nq);
          q_[6] = 1.0;
          q_dot_.setZero(pin_model_.nv);
          ff_torque.setZero(pin_model_.nv - 6);

          for (size_t i = 1; i < pin_model_.joints.size(); ++i) {
            if (pin_model_.joints[i].nv() == 1) {
              pin_joint_names_.push_back(pin_model_.names[i]);
            }
          }

          // FIX: Map target_pos (ordered by parameter) to q_target_ (ordered by
          // URDF)

          target_poses[0] = Eigen::VectorXd::Zero(pin_joint_names_.size());
          target_poses[1] = Eigen::VectorXd::Zero(pin_joint_names_.size());
          for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
            auto it = std::find(joint_names_.begin(), joint_names_.end(),
                                pin_joint_names_[i]);
            if (it != joint_names_.end()) {
              size_t param_idx = std::distance(joint_names_.begin(), it);
              target_poses[0][i] = target_pos1[param_idx];
              target_poses[1][i] = target_pos2[param_idx];

            } else {
              RCLCPP_WARN(get_node()->get_logger(),
                          "Joint %s not found in parameter list!",
                          pin_joint_names_[i].c_str());
            }
          }

          urdf_received_ = true;
        }
      });

  RCLCPP_INFO(get_node()->get_logger(), "StandUp configuration done!");
  return controller_interface::CallbackReturn::SUCCESS;
}
controller_interface::CallbackReturn
DownThenStand::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "Starting activation of StandUp");
  auto start_time = get_node()->now();
  rclcpp::Duration timeout = rclcpp::Duration::from_seconds(5.0);

  while (!urdf_received_ && (get_node()->now() - start_time) < timeout) {
    RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(),
                         1000, "Waiting for URDF on /robot_description...");
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }

  if (!urdf_received_) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Timed out waiting for URDF on /robot_description.");
    return controller_interface::CallbackReturn::FAILURE;
  }

  cmd_map_.clear();
  for (auto &interface : command_interfaces_) {
    cmd_map_[interface.get_name()] = &interface;
  }

  state_map_.clear();
  for (auto &interface : state_interfaces_) {
    state_map_[interface.get_name()] = &interface;
  }

  update_joint_states();

  done_ = false;
  q_init_ = q_.tail(12);
  start_time = get_node()->now();
  start_time_ = start_time;
  current_phase = StandUpPhase::Down;
  RCLCPP_INFO(get_node()->get_logger(),
              "StandUp activated. Starting Pure Damping Phase (1s)...");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
DownThenStand::command_interface_configuration() const {

  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto &joint : joint_names_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
    config.names.push_back(joint + "/" +
                           hardware_interface::HW_IF_PROPORTIONAL_GAIN);
    config.names.push_back(joint + "/" +
                           hardware_interface::HW_IF_DERIVATIVE_GAIN);
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::InterfaceConfiguration
DownThenStand::state_interface_configuration() const {

  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto &joint : joint_names_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::CallbackReturn
DownThenStand::on_deactivate(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_node()->get_logger(), "StandUp deactivated.");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type
DownThenStand::update(const rclcpp::Time &time,
                      const rclcpp::Duration & /*period*/) {
  update_joint_states();

  double elapsed = (time - start_time_).seconds();

  Vector12d *current_target = &q_init_;
  if (elapsed > duration_ && current_phase == StandUpPhase::Down) {
    current_phase = StandUpPhase::Stand;
    RCLCPP_INFO(get_node()->get_logger(),"Standing Up");
    start_time_ = time;
    q_init_ = q_.tail(12);
  }

  if (current_phase == StandUpPhase::Down) {
    current_target = &target_poses[0];
  } else if (current_phase == StandUpPhase::Stand) {
    current_target = &target_poses[1];
  }

  pinocchio::computeGeneralizedGravity(pin_model_, *pin_data_, q_);

  double ratio = std::clamp(elapsed / duration_, 0.0, 1.0);
  Vector12d control_target =
      smooth_interpolation(q_init_, *current_target, ratio);

  ff_torque = pin_data_->g.tail(12);
  send_command_to_hardware(control_target);
  return controller_interface::return_type::OK;
}

DownThenStand::Vector12d
DownThenStand::smooth_interpolation(const Vector12d &q_start,
                                    const Vector12d &q_end, double ratio) {
  double smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio);
  return (1.0 - smooth_ratio) * q_start + smooth_ratio * q_end;
}

void DownThenStand::update_joint_states() {
  for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
    q_[i + 7] =
        state_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_POSITION)
            ->get_optional()
            .value();
    q_dot_[i + 6] =
        state_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_VELOCITY)
            ->get_optional()
            .value();
  }
}

void DownThenStand::send_command_to_hardware(
    const Vector12d &desired_position) {

  for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
    std::ignore =
        cmd_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_EFFORT)
            ->set_value(ff_torque[i]);
    std::ignore = cmd_map_
                      .at(pin_joint_names_[i] + "/" +
                          hardware_interface::HW_IF_PROPORTIONAL_GAIN)
                      ->set_value(kp_);
    std::ignore = cmd_map_
                      .at(pin_joint_names_[i] + "/" +
                          hardware_interface::HW_IF_DERIVATIVE_GAIN)
                      ->set_value(kd_);
    std::ignore =
        cmd_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_POSITION)
            ->set_value(desired_position[i]);
    std::ignore =
        cmd_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_VELOCITY)
            ->set_value(0.);
  }
}

} // namespace stand_up_pkg

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(stand_up_pkg::DownThenStand,
                       controller_interface::ControllerInterface)