#include "whole_body_controller/whole_body_controller.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "wbc_control_msgs/msg/foot_contact_flags.hpp"
#include "wbc_control_msgs/msg/mpc_trajectory.hpp"
#include "whole_body_controller/optimizer.hpp"

#include <controller_interface/controller_interface_base.hpp>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "whole_body_controller/dynamic_system.hpp"
#include "whole_body_controller/system_state.hpp"

#include <cstddef>
#include <hardware_interface/loaned_command_interface.hpp>

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/time.hpp>
#include <sstream>
#include <tf2_eigen/tf2_eigen.hpp>

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace wbc {

// Print helper for fixed-width header bars
inline std::string makeDivider(int width = 72, char symbol = '-') {
  return std::string(width, symbol) + "\n";
}
std::string log_control(const ControlOutput &target) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(3);
  ss << makeDivider(72, '=');
  // ss << " [KINEMATICS]\n";
  // ss << "  Base Pos (x,y,z)   : [ " << std::setw(7) << target.q(0) << ", "
  //    << std::setw(7) << target.q(1) << ", " << std::setw(7) << target.q(2)
  //    << " ] m\n";
  // ss << "  Base Orient (w,x,y,z): [ " << std::setw(7) << target.q(6) << ", "
  //    << std::setw(7) << target.q(3) << ", " << std::setw(7) << target.q(4)
  //    << ", " << std::setw(7) << target.q(5) << " ] (Quat)\n";
  // ss << "\n  Joint Pos (1-6)   : ";
  // for (int i = 7; i < 13; ++i)
  //   ss << std::setw(7) << target.q(i) << " ";
  // ss << "\n  Joint Pos (7-12)   : ";
  // for (int i = 13; i < 19; ++i)
  //   ss << std::setw(7) << target.q(i) << " ";
  ss << "  Leg |     Contact Force (N)     Torque (Nm)\n";
  ss << "------|------------------------|------------------------\n";

  const std::array<std::string, 4> leg_names = {"FR", "FL", "RR", "RL"};
  for (size_t i = 0; i < 4; ++i) {
    ss << "  " << leg_names[i] << "  | [" << std::setw(5)
       << target.contact_forces[3 * i] << ", " << std::setw(5)
       << target.contact_forces[3 * i + 1] << ", " << std::setw(5)
       << target.contact_forces[3 * i + 2] << " ]  |   [" << std::setw(5)
       << target.torque[3 * i] << ", " << std::setw(5)
       << target.torque[3 * i + 1] << ", " << std::setw(5)
       << target.torque[3 * i + 2] << " ]\n";
  }
  return ss.str();
}
// Function to format SystemState into a tabular string
std::string log_system_state(const SystemState &state) {
  std::ostringstream ss;

  // Fixed floating point precision for clean column alignment
  ss << std::fixed << std::setprecision(3);

  ss << makeDivider(72, '=');
  ss << "                          SYSTEM STATE REPORT                         "
        "\n";
  ss << makeDivider(72, '=');

  // 1. KINEMATICS (Generalized Coordinates)
  ss << " [KINEMATICS]\n";
  ss << "  Base Pos (x,y,z)   : [ " << std::setw(7) << state.q(0) << ", "
     << std::setw(7) << state.q(1) << ", " << std::setw(7) << state.q(2)
     << " ] m\n";
  ss << "  Base Orient (w,x,y,z): [ " << std::setw(7) << state.q(6) << ", "
     << std::setw(7) << state.q(3) << ", " << std::setw(7) << state.q(4) << ", "
     << std::setw(7) << state.q(5) << " ] (Quat)\n";

  ss << "  Joint Pos  (1-6)   : ";
  for (int i = 7; i < 13; ++i)
    ss << std::setw(7) << state.q(i) << " ";
  ss << "\n  Joint Pos (7-12)   : ";
  for (int i = 13; i < 19; ++i)
    ss << std::setw(7) << state.q(i) << " ";
  ss << "\n" << makeDivider(72, '-');

  // 2. HARDWARE STATE (Torques & Gains by Joint Index)
  ss << " [HARDWARE STATE]\n";
  ss << "  Joint | Torque (Nm) \n";
  ss << "  ------|-------------\n";
  for (int i = 0; i < 12; ++i) {
    ss << "   " << std::setw(2) << i << "   | " << std::setw(11)
       << state.j_torq(i) << " |\n";
  }
  ss << makeDivider(72, '-');

  // 3. FOOT STATES
  ss << " [FOOT STATES]\n";
  ss << "  Leg | Contact |      Position (m)      |     Velocity (m/s)     |   "
        " Contact Force (N)   \n";
  ss << "  "
        "----|---------|------------------------|------------------------|-----"
        "-------------------\n";

  const std::array<std::string, 4> leg_names = {"FL", "FR", "RL", "RR"};
  for (size_t i = 0; i < state.fs.size(); ++i) {
    const auto &f = state.fs[i];
    ss << "   " << std::setw(2) << leg_names[i] << " | " << std::setw(7)
       << (f.contact ? "  YES  " : "   NO  ") << " | " << "[ " << std::setw(5)
       << f.foot_pos.x() << ", " << std::setw(5) << f.foot_pos.y() << ", "
       << std::setw(5) << f.foot_pos.z() << " ] | " << "[ " << std::setw(5)
       << f.foot_vel.x() << ", " << std::setw(5) << f.foot_vel.y() << ", "
       << std::setw(5) << f.foot_vel.z() << " ]\n";
  }
  ss << makeDivider(72, '=');

  return ss.str();
}

controller_interface::CallbackReturn WBC::on_init() {

  // optimizer_ = std::make_unique<Optimizer>();
  // dynamic_system_ = std::make_unique<DynamicSystem>();
  try {
    // Declare parameters, e.g., joint names, QP weights, friction mu
    auto_declare<std::vector<std::string>>("joint_suffixes",
                                           std::vector<std::string>());
    auto_declare<std::vector<std::string>>("legs_prefixes",
                                           std::vector<std::string>());
    auto_declare<std::string>("foot_suffix", "_foot");
    auto_declare<std::string>("hip_suffix", "_hip");
    auto_declare<std::string>("thigh_suffix", "_thigh");
    auto_declare<std::string>("calf_suffix", "_calf");

    auto_declare<std::vector<std::string>>("base_state_int",
                                           std::vector<std::string>());

  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Exception thrown during on_init: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
WBC::on_configure(const rclcpp_lifecycle::State & /*previous_state*/) {

  joint_suffixes_ =
      get_node()->get_parameter("joint_suffixes").as_string_array();
  leg_prefixes_ = get_node()->get_parameter("legs_prefixes").as_string_array();
  base_state_int_ =
      get_node()->get_parameter("base_state_int").as_string_array();

  hip_suffix_ = get_node()->get_parameter("hip_suffix").as_string();
  thigh_suffix_ = get_node()->get_parameter("thigh_suffix").as_string();
  calf_suffix_ = get_node()->get_parameter("calf_suffix").as_string();

  foot_suffix_ = get_node()->get_parameter("foot_suffix").as_string();

  if (joint_suffixes_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "'joint_suffixes' parameter is empty!");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (leg_prefixes_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "'legs_prefixes' parameter is empty!");
    return controller_interface::CallbackReturn::ERROR;
  }

  /////////////////////////////////////////////////////////////////////////////////////
  mpc_trj_sub =
      get_node()->create_subscription<wbc_control_msgs::msg::MpcTrajectory>(
          "/mpc_trajectory", rclcpp::QoS(10),
          [this](const wbc_control_msgs::msg::MpcTrajectory msg) {
            trajectory_update(msg);
          });
  fc_pub =
      get_node()->create_publisher<wbc_control_msgs::msg::FootContactFlags>(
          "/foot_contacts", rclcpp::QoS(1));
  ////////////////////////////////////////////////////////////////////////////////////

  urdf_sub_ = get_node()->create_subscription<std_msgs::msg::String>(
      "/robot_description", rclcpp::QoS(1).transient_local(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (urdf_received_) {
          return;
        }

        std::string base_name = "base";
        std::string urdf = msg->data;

        std::vector<std::string> hip_names;
        std::vector<std::string> thigh_names;
        std::vector<std::string> calf_names;
        std::vector<std::string> feet_names;

        std::vector<std::string> joints_names;

        for (const auto &leg : leg_prefixes_) {
          hip_names.push_back(leg + hip_suffix_);
          thigh_names.push_back(leg + thigh_suffix_);
          calf_names.push_back(leg + calf_suffix_);
          feet_names.push_back(leg + foot_suffix_);
          for (const auto &joint : joint_suffixes_) {
            joints_names.push_back(leg + joint);
          }
        }

        if (!dynamic_system_.initialize(urdf, base_name, hip_names, thigh_names,
                                        calf_names, feet_names, joints_names)) {
          RCLCPP_FATAL(get_node()->get_logger(),
                       "Pinocchio model initialization failed");
          return;
        }
        urdf_received_ = true;
      });
  system_state.foot_jacobian.setZero();
  system_state.base_jacobian.setZero();
  system_state.j_torq.setZero();
  system_state.hu.setZero();

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
WBC::command_interface_configuration() const {

  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto &leg : leg_prefixes_) {
    for (const auto &joint : joint_suffixes_) {
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_EFFORT);
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_POSITION);
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_VELOCITY);
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_PROPORTIONAL_GAIN);
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_DERIVATIVE_GAIN);
    }
  }
  return config;
}

controller_interface::InterfaceConfiguration
WBC::state_interface_configuration() const {

  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // claim leg state interfaces
  for (const auto &leg : leg_prefixes_) {
    for (const auto &joint : joint_suffixes_) {
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_EFFORT);
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_POSITION);
      config.names.push_back(leg + joint + "/" +
                             hardware_interface::HW_IF_VELOCITY);
    }
  }
  // Claim Floating Base State from the Estimator
  for (const auto &var : base_state_int_) {
    config.names.push_back("state_estimator/" + var);
  }
  return config;
}

controller_interface::CallbackReturn
WBC::on_deactivate(const rclcpp_lifecycle::State &) {
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
WBC::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
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
  /////////////////////////////////////////////////////////////////////////
  // store state interfaces in maps
  std::unordered_map<std::string, hardware_interface::LoanedStateInterface *>
      st_map;
  std::unordered_map<std::string, hardware_interface::LoanedCommandInterface *>
      cm_map;

  for (auto &interface : state_interfaces_) {
    st_map[interface.get_name()] = &interface;
  }
  for (auto &interface : command_interfaces_) {
    cm_map[interface.get_name()] = &interface;
  }
  for (size_t i = 0; i < base_state_int_.size(); ++i) {
    BaseIfs bi = static_cast<BaseIfs>(i);
    base_interfaces_names[bi] = "state_estimator/" + base_state_int_[i];
    base_interfaces_map[bi] = st_map[base_interfaces_names[bi]];
  }

  std::vector<std::string> actuator_interfaces = {
      hardware_interface::HW_IF_POSITION,
      hardware_interface::HW_IF_VELOCITY,
      hardware_interface::HW_IF_EFFORT,
      hardware_interface::HW_IF_PROPORTIONAL_GAIN,
      hardware_interface::HW_IF_DERIVATIVE_GAIN,

  };

  // construct what the names are supposed to be
  for (size_t i = 0; i < leg_prefixes_.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {   // 3 actuators per leg
      for (size_t k = 0; k < 5; ++k) { // 5 interfaces per actuator
        // 3 state interfaces per act
        Legs l = static_cast<Legs>(i);
        Actuator a = static_cast<Actuator>(j);
        ActIfs ai = static_cast<ActIfs>(k);
        std::string name = leg_prefixes_[i] + joint_suffixes_[j] + "/" +
                           actuator_interfaces[k];
        leg_map_names_[l][a][ai] = name;
        leg_state_interfaces_map[l][a][ai] = st_map[name];
        leg_command_interfaces_map[l][a][ai] = cm_map[name];
      }
    }
  }

  if (!optimizer_.init()) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Optimizer wasnt initialized correctly!");
    return controller_interface::CallbackReturn::FAILURE;
  }
  system_state.q.setZero();
  system_state.q[6] = 1.;
  system_state.q_dot.setZero();

  __first_update = true;
  return controller_interface::CallbackReturn::SUCCESS;
}
// callback
void WBC::trajectory_update(const wbc_control_msgs::msg::MpcTrajectory &msg) {
  std::vector<TargetState> sys;
  std::stringstream sis;
  for (auto &target : msg.steps) {
    TargetState ss;
    // base
    ss.q.topRows(7) << target.body_pose.position.x, target.body_pose.position.y,
        target.body_pose.position.z, target.body_pose.orientation.x,
        target.body_pose.orientation.y, target.body_pose.orientation.z,
        target.body_pose.orientation.w;

    ss.q_dot.topRows(6) << target.body_velocity.linear.x,
        target.body_velocity.linear.y, target.body_velocity.linear.z,
        target.body_velocity.angular.x, target.body_velocity.angular.y,
        target.body_velocity.angular.z;

    ss.q_ddot.topRows(6) << target.body_acceleration.linear.x,
        target.body_acceleration.linear.y, target.body_acceleration.linear.z,
        target.body_acceleration.angular.x, target.body_acceleration.angular.y,
        target.body_acceleration.angular.z;
    // feet
    ss.q_ddot.bottomRows(12).setZero();
    int id = 0;
    for (auto &foot : target.feet) {
      ss.fs[id].foot_pos << foot.position.x, foot.position.y, foot.position.z;
      ss.fs[id].foot_vel << foot.velocity.x, foot.velocity.y, foot.velocity.z;
      ss.contact_forces.middleRows(3 * id, 3) << foot.contact_force.x,
          foot.contact_force.y, foot.contact_force.z;
      ss.fs[id].contact = foot.is_contact;
      ++id;
    };
    ss.j_kp.setOnes();
    ss.j_kd.setOnes();
    ss.j_kp *= 60.;
    ss.j_kd *= 2.4;

    // legs(target)
    dynamic_system_.inverse_kinematics(ss);
    sys.push_back(ss);
  }

  sis << sys.back() << "\n";
  // RCLCPP_INFO(get_node()->get_logger(),"%zu last target:\n%s",sys.size(),
  // sis.str().c_str());

  interpolator_.set_target_from_non_rt(sys);
}

void WBC::update_system_state() {
  system_state.q.topRows(7)
      << base_interfaces_map.at(BaseIfs::POS_X)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::POS_Y)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::POS_Z)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::Q_X)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::Q_Y)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::Q_Z)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::Q_W)->get_optional().value_or(1.);

  system_state.q_dot.topRows(6)
      << base_interfaces_map.at(BaseIfs::VEL_X)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::VEL_Y)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::VEL_Z)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::RVEL_X)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::RVEL_Y)->get_optional().value_or(0.),
      base_interfaces_map.at(BaseIfs::RVEL_Z)->get_optional().value_or(0.);

  int num_of_legs = leg_state_interfaces_map.size();
  int num_of_act = leg_state_interfaces_map.at(Legs::LF).size();

  for (int i = 0; i < num_of_legs; ++i)
    for (int j = 0; j < num_of_act; ++j) {
      auto &actuator = leg_state_interfaces_map.at(static_cast<Legs>(i))
                           .at(static_cast<Actuator>(j));
      system_state.q[7 + i * num_of_act + j] =
          actuator.at(ActIfs::POS)->get_optional().value_or(0.);

      system_state.q_dot[6 + i * num_of_act + j] =
          actuator.at(ActIfs::VEL)->get_optional().value_or(0.);

      system_state.j_torq[i * num_of_act  + j] =
          actuator.at(ActIfs::TRQ)->get_optional().value_or(0.);
    }
  dynamic_system_.update_dynamics(system_state);
}

void WBC::send_system_state(const TargetState &target,
                            const ControlOutput &output) {
  int num_of_legs = leg_command_interfaces_map.size();
  int n_o_a = leg_command_interfaces_map.at(Legs::LF).size();
  auto q = target.q.tail<12>();
  auto q_dot = target.q_dot.tail<12>();

  for (int i = 0; i < num_of_legs; ++i)
    for (int j = 0; j < n_o_a; ++j) {
      auto &coms = leg_command_interfaces_map.at(static_cast<Legs>(i))
                       .at(static_cast<Actuator>(j));

      (void)coms.at(ActIfs::POS)->set_value(q[i * n_o_a + j]);
      (void)coms.at(ActIfs::VEL)->set_value(q_dot[i * n_o_a + j]);
      (void)coms.at(ActIfs::TRQ)->set_value(output.torque[i * n_o_a + j]);
      (void)coms.at(ActIfs::SKP)->set_value(target.j_kp[i * n_o_a + j]);
      (void)coms.at(ActIfs::SKD)->set_value(target.j_kd[i * n_o_a + j]);
    }
}

controller_interface::return_type
WBC::update(const rclcpp::Time & /*time*/,
            const rclcpp::Duration & /*period*/) {

  update_system_state();
  if (__first_update) {
    // clang-format off
    InterpolatrorConfig config = {
      10,
         100, 
   10,
      10
    };
    // clang-format on
    double fz_target = dynamic_system_.get_total_mass() * 9.81 / 4.;
    TargetState default_target;
    default_target.contact_forces.setZero();
    default_target.contact_forces << 0.0, 0.0, fz_target, 0.0, 0.0, fz_target,
        0.0, 0.0, fz_target, 0.0, 0.0, fz_target;
    // default_target.fs = system_state.fs;
    default_target.q.setZero();
    default_target.q[2] = 0.3;
    default_target.q[6] = 1.0;
    default_target.q_dot.setZero();
    default_target.q_ddot.setZero();

    for (int i = 0; i < 4; ++i) {
      system_state.fs[i].contact = true;
      default_target.fs[i].contact = true;
      default_target.fs[i].foot_vel.setZero();
    }
    default_target.fs[0].foot_pos << 0.19, -0.142, 0.0;
    default_target.fs[1].foot_pos << 0.19, 0.142, 0.0;
    default_target.fs[2].foot_pos << -0.19, -0.142, 0.0;
    default_target.fs[3].foot_pos << -0.19, 0.142, 0.0;

    default_target.j_kp.setOnes();
    default_target.j_kd.setOnes();
    default_target.j_kp *= 60.;
    default_target.j_kd *= 2.4;

    dynamic_system_.inverse_kinematics(default_target);
    interpolator_.init(dynamic_system_.get_model(), config, default_target);
    __first_update = false;
  }
  auto target = interpolator_.get_next_target();
  // update foot contacts for state estimation:
  wbc_control_msgs::msg::FootContactFlags msg;
  msg.fr = system_state.fs[0].contact;
  msg.fl = system_state.fs[1].contact;
  msg.rl = system_state.fs[2].contact;
  msg.rr = system_state.fs[3].contact;
  fc_pub->publish(msg);

  ControlOutput output;
  if (!optimizer_.solve(system_state, target, output)) {
    RCLCPP_FATAL_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(),
                          100, "Solver failed!");
  }
  dynamic_system_.inverse_dynamics(system_state, output);
  send_system_state(target, output);

  std::stringstream ss;
  ss << "-----------------System--------------------\n"
     << system_state.fs[0] << "\n"
     << system_state.fs[1] << "\n"
     << system_state.fs[2] << "\n"
     << system_state.fs[3] << "\n"
     << "\n";
  ss << "--------------State Difference------------\n "
     << (target.q - system_state.q).transpose() << "\n";
  // ss << "-----------------Output-------------------\n " << output << "\n";

  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 100,
                       "\n%s", ss.str().c_str());

  return controller_interface::return_type::OK;
}

} // namespace wbc
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(wbc::WBC, controller_interface::ControllerInterface)