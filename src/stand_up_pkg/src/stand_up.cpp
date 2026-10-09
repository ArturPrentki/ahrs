#include "stand_up_pkg/stand_up.hpp"
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

controller_interface::CallbackReturn StandUp::on_init() {
  try {
    auto_declare<std::vector<std::string>>("joints",
                                           std::vector<std::string>());
    // Update default target targets to FR, FL, RR, RL order
    auto_declare<std::vector<double>>("target_stand",
                                      {-0.1, 0.8, -1.5,  // FR (HAA, HFE, KFE)
                                       0.1, 0.8, -1.5,   // FL
                                       -0.1, 1.0, -1.5,  // RR
                                       0.1, 1.0, -1.5}); // RL
    auto_declare<std::vector<double>>("target_sit", {-0.01, 1.27, -2.8, ///
                                                     0.01, 1.27, -2.8,  //
                                                     -0.3, 1.31, -2.8,  //
                                                     0.3, 1.31, -2.8});

    auto_declare<double>("kp", 250.0);
    auto_declare<double>("kd", 20.0);
    auto_declare<double>("duration", 3.0);

  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Exception thrown during on_init: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
StandUp::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "Starting configuration of StandUp");
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  std::vector<double> target_pos1 =
      get_node()->get_parameter("target_sit").as_double_array();
  std::vector<double> target_pos2 = get_node()->get_parameter("target_stand").as_double_array();
  kp_ = get_node()->get_parameter("kp").as_double();
  kd_ = get_node()->get_parameter("kd").as_double();
  duration_ = get_node()->get_parameter("duration").as_double();

  if (joint_names_.empty() || target_pos1.size() <12||target_pos2.size() < 12) {
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
      [this, target_pos1, target_pos2](const std_msgs::msg::String::SharedPtr msg) {
        if (!urdf_received_) {
          std::string base_name = "trunk";

          pinocchio::urdf::buildModelFromXML(
              msg->data, pinocchio::JointModelFreeFlyer(), pin_model_);

          pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
          q_.setZero(pin_model_.nq);
          q_[6] = 1.0;
          q_dot_.setZero(pin_model_.nv);
          tau_cmd_.setZero(pin_model_.nv - 6);

          for (size_t i = 1; i < pin_model_.joints.size(); ++i) {
            if (pin_model_.joints[i].nv() == 1) {
              pin_joint_names_.push_back(pin_model_.names[i]);
            }
          }

          // FIX: Map target_pos (ordered by parameter) to q_target_ (ordered by
          // URDF)
          
          q_targets_[0] = Eigen::VectorXd::Zero(pin_joint_names_.size());
          q_targets_[1] = Eigen::VectorXd::Zero(pin_joint_names_.size());
          for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
            auto it = std::find(joint_names_.begin(), joint_names_.end(),
                                pin_joint_names_[i]);
            if (it != joint_names_.end()) {
              size_t param_idx = std::distance(joint_names_.begin(), it);
              q_targets_[0][i] = target_pos1[param_idx];
              q_targets_[1][i] = target_pos2[param_idx];
              
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
StandUp::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
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

  RCLCPP_INFO(get_node()->get_logger(),
              "StandUp activated. Starting Pure Damping Phase (1s)...");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
StandUp::command_interface_configuration() const {

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
StandUp::state_interface_configuration() const {

  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto &joint : joint_names_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::CallbackReturn
StandUp::on_deactivate(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_node()->get_logger(), "StandUp deactivated.");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type
StandUp::update(const rclcpp::Time &time, const rclcpp::Duration & /*period*/) {

  double elapsed = (time - start_time_).seconds();
  update_joint_states();

  constexpr double damping_duration =
      1.010; // Pure damping phase duration (seconds)

  if (elapsed < damping_duration) {
    // 1. Pure Damping Phase
    tau_cmd_ = kd_ * (-q_dot_.tail(12));

    // Update q_init_ continuously so interpolation starts from
    // the robot's actual settled state at t = 1.0s
    q_init_ = q_.tail(12);
  } else {
    // 2. Stand-Up Interpolation Phase
    double homing_elapsed = elapsed - damping_duration;
    double ratio = std::clamp(homing_elapsed / duration_, 0.0, 1.0);

    // Smooth trajectory from q_init_ to q_target_
    Eigen::VectorXd q_des = smooth_interpolation(q_init_, q_target_, ratio);
    pinocchio::computeGeneralizedGravity(pin_model_, *pin_data_, q_);

    tau_cmd_ = kp_ * (q_des - q_.tail(12)) + kd_ * (-q_dot_.tail(12)) +
               pin_data_->g.tail(12);
  }

  send_command_to_hardware();
  return controller_interface::return_type::OK;
}

Eigen::VectorXd StandUp::smooth_interpolation(const Eigen::VectorXd &q_start,
                                              const Eigen::VectorXd &q_end,
                                              double ratio) {
  double smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio);
  return (1.0 - smooth_ratio) * q_start + smooth_ratio * q_end;
}

void StandUp::update_joint_states() {
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

void StandUp::send_command_to_hardware() {

  for (size_t i = 0; i < pin_joint_names_.size(); ++i) {
    std::ignore =
        cmd_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_EFFORT)
            ->set_value(tau_cmd_[i]);
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
            ->set_value(q_target_[i]);
    std::ignore =
        cmd_map_
            .at(pin_joint_names_[i] + "/" + hardware_interface::HW_IF_VELOCITY)
            ->set_value(0.);
  }
}

} // namespace stand_up_pkg

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(stand_up_pkg::StandUp,
                       controller_interface::ControllerInterface)