#pragma once
#include "wbc_control_msgs/msg/foot_contact_flags.hpp"
#include "whole_body_controller/dynamic_system.hpp"
#include "whole_body_controller/optimizer.hpp"
#include "whole_body_controller/control_interpolator.hpp"
#include <controller_interface/controller_interface.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>

#include <rclcpp/publisher.hpp>
#include <string>
#include <unordered_map>

#include "std_msgs/msg/string.hpp"
#include "wbc_control_msgs/msg/mpc_trajectory.hpp"
#include "whole_body_controller/system_state.hpp"

#include <realtime_tools/realtime_buffer.hpp>
namespace wbc {

class WBC : public controller_interface::ControllerInterface {
  enum class BaseIfs : int {
    POS_X = 0,
    POS_Y,
    POS_Z,
    Q_X,
    Q_Y,
    Q_Z,
    Q_W,
    VEL_X,
    VEL_Y,
    VEL_Z,
    RVEL_X,
    RVEL_Y,
    RVEL_Z,
  };
  enum class ActIfs : int {
    POS = 0,
    VEL,
    TRQ,
    SKP,
    SKD,
  };

  enum class Actuator : int { HIP = 0, THIGH, CALF };
  enum class Legs : int { LF = 0, RF, LR, RR };
  using map_actuators_names = std::unordered_map<ActIfs, std::string>;
  using map_leg_names = std::unordered_map<Actuator, map_actuators_names>;
  using map_legs_names = std::unordered_map<Legs, map_leg_names>;

  using map_actuators_state =
      std::unordered_map<ActIfs, hardware_interface::LoanedStateInterface *>;
  using map_leg_state = std::unordered_map<Actuator, map_actuators_state>;
  using map_legs_states = std::unordered_map<Legs, map_leg_state>;

  using map_actuators_command =
      std::unordered_map<ActIfs, hardware_interface::LoanedCommandInterface *>;
  using map_leg_command = std::unordered_map<Actuator, map_actuators_command>;
  using map_legs_commands = std::unordered_map<Legs, map_leg_command>;

public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  controller_interface::return_type
  update(const rclcpp::Time &time, const rclcpp::Duration &period) override;

  controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr urdf_sub_;
  bool urdf_received_ = false;

  rclcpp::Subscription<wbc_control_msgs::msg::MpcTrajectory>::SharedPtr
      mpc_trj_sub;
  void trajectory_update(const wbc_control_msgs::msg::MpcTrajectory& msg);
  rclcpp::Publisher<wbc_control_msgs::msg::FootContactFlags>::SharedPtr fc_pub;

  std::unordered_map<BaseIfs, std::string> base_interfaces_names;
  std::unordered_map<BaseIfs, hardware_interface::LoanedStateInterface *>
      base_interfaces_map;

  map_legs_names leg_map_names_;
  map_legs_states leg_state_interfaces_map;
  map_legs_commands leg_command_interfaces_map;

  DynamicSystem dynamic_system_;
  Optimizer optimizer_;
  ControlInterpolator interpolator_;
  SystemState system_state; // what does system do currently

  std::vector<std::string> leg_prefixes_;
  std::vector<std::string> joint_suffixes_;
  std::string foot_suffix_;
  std::string hip_suffix_;
  std::string thigh_suffix_;
  std::string calf_suffix_;
  
  std::vector<std::string> base_state_int_;

  bool __first_update;
  
  void update_system_state();
  void send_system_state(const TargetState &target, const ControlOutput& output);
};
} // namespace wbc