#include "quadruped_estimators/contact_inekf_estimator.hpp"
#include "InEKF.h"
#include <array>
#include <memory>
#include <rclcpp/logging.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <utility>
namespace quadruped_estimators {

controller_interface::InterfaceConfiguration
ContactInEKFEstimator::command_interface_configuration() const {
  return {controller_interface::interface_configuration_type::NONE};
}

controller_interface::InterfaceConfiguration
ContactInEKFEstimator::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Request IMU data from hardware
  config.names.push_back("imu_sensor/linear_acceleration.x");
  config.names.push_back("imu_sensor/linear_acceleration.y");
  config.names.push_back("imu_sensor/linear_acceleration.z");
  config.names.push_back("imu_sensor/angular_velocity.x");
  config.names.push_back("imu_sensor/angular_velocity.y");
  config.names.push_back("imu_sensor/angular_velocity.z");

  std::vector<std::string> joint_names = {
      "FL_hip_joint",   "FL_thigh_joint", "FL_calf_joint",  "FR_hip_joint",
      "FR_thigh_joint", "FR_calf_joint",  "RL_hip_joint",   "RL_thigh_joint",
      "RL_calf_joint",  "RR_hip_joint",   "RR_thigh_joint", "RR_calf_joint"};
  for (const auto &joint : joint_names) {
    config.names.push_back(joint + "/position");
  }

  return config;
}

controller_interface::CallbackReturn ContactInEKFEstimator::on_init() {
  filter_ = std::shared_ptr<inekf::InEKF>(new inekf::InEKF());
  urdf_received_ = false;
  base_state_.resize(13, 0.0);
  base_state_[6] = 1.0; // quat_w default

  // Initialize InEKF Noise Parameters (Tune these for your specific robot/IMU)
  inekf::NoiseParams noise_params;
  noise_params.setGyroscopeNoise(0.01);
  noise_params.setAccelerometerNoise(0.1);
  noise_params.setGyroscopeBiasNoise(0.00001);
  noise_params.setAccelerometerBiasNoise(0.0001);
  noise_params.setContactNoise(0.01);

  // Set initial state
  inekf::RobotState initial_state;
  // // (Optional) Configure initial pose, covariance here if needed
  Eigen::Vector3d init_pos(0.0, 0.0, 0.30);
  initial_state.setPosition(init_pos);
  filter_->setState(initial_state);
  filter_->setNoiseParams(noise_params);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
ContactInEKFEstimator::on_configure(const rclcpp_lifecycle::State &) {

  std::array<bool, 4> contacts = {true, true, true, true};

  command_contacts.writeFromNonRT(contacts);

  auto wbc_command_qos = rclcpp::SensorDataQoS();
  wbc_command_qos.keep_last(1);
  foot_contacts_sub =
      get_node()->create_subscription<wbc_control_msgs::msg::FootContactFlags>(
          "/foot_contacts", wbc_command_qos,
          [this](const wbc_control_msgs::msg::FootContactFlags::SharedPtr msg) {
            std::array<bool, 4> contacts = {msg->fl, msg->fr, msg->rl, msg->rr};
            command_contacts.writeFromNonRT(contacts);
          });

  std::string urdf_string;
  urdf_sub_ = get_node()->create_subscription<std_msgs::msg::String>(
      "/robot_description", rclcpp::QoS(1).transient_local(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (urdf_received_) {
          return;
        }
        RCLCPP_INFO(get_node()->get_logger(), "Got URDF file!");
        pin_model_ = std::make_shared<pinocchio::Model>();
        // 2. Build Pinocchio model from URDF XML string
        pinocchio::urdf::buildModelFromXML(
            msg->data, pinocchio::JointModelFreeFlyer(), *pin_model_);
        pin_data_ = std::make_shared<pinocchio::Data>(*pin_model_);
        urdf_received_ = true;
      });

  odom_pub_ = get_node()->create_publisher<nav_msgs::msg::Odometry>(
      "/odom", rclcpp::QoS(1).transient_local());

  return controller_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
ContactInEKFEstimator::on_export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> interfaces;
  std::vector<std::string> names = {
      "pos_x",     "pos_y",     "pos_z",     "quat_x",    "quat_y",
      "quat_z",    "quat_w",    "lin_vel_x", "lin_vel_y", "lin_vel_z",
      "ang_vel_x", "ang_vel_y", "ang_vel_z"};
  for (size_t i = 0; i < names.size(); ++i) {
    interfaces.emplace_back(hardware_interface::StateInterface(
        get_node()->get_name(), names[i], &base_state_[i]));
  }
  return interfaces;
}

bool ContactInEKFEstimator::on_set_chained_mode(bool /*chained_mode*/) {
  return true;
}

controller_interface::return_type
ContactInEKFEstimator::update_reference_from_subscribers(
    const rclcpp::Time &, const rclcpp::Duration &) {
  // Can be left empty, as we handle RT logic in update_and_write_commands
  return controller_interface::return_type::OK;
}

controller_interface::return_type
ContactInEKFEstimator::update_and_write_commands(
    const rclcpp::Time &time, const rclcpp::Duration &period) {

  double dt = period.seconds();

  // 1. Read IMU from state_interfaces_
  // (Assuming IMU is at indices 0-5 based on state_interface_configuration)
  Eigen::Vector3d acc(state_interfaces_[0].get_optional().value(),
                      state_interfaces_[1].get_optional().value(),
                      state_interfaces_[2].get_optional().value());

  Eigen::Vector3d gyro(state_interfaces_[3].get_optional().value(),
                       state_interfaces_[4].get_optional().value(),
                       state_interfaces_[5].get_optional().value());

  // 2. Propagation Step (IMU)
  Eigen::Matrix<double, 6, 1> imu_meas;
  imu_meas << gyro, acc;
  filter_->Propagate(imu_meas, dt);

  // Read Joint Positions (Indices 6 to 17)
  std::vector<double> joint_positions;
  for (int i = 6; i < 18; i++) {
    joint_positions.push_back(state_interfaces_[i].get_optional().value());
  }

  inekf::vectorKinematics kinematics_list;
  std::vector<std::pair<int, bool>> contacts;
  Eigen::Matrix<double, 6, 6> contact_cov =
      Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
  auto contacts_ptr = command_contacts.readFromRT();
  std::array<bool, 4> contact_flags =
      contacts_ptr ? *contacts_ptr
                   : std::array<bool, 4>{true, true, true, true};
  for (int leg_id = 0; leg_id < 4; leg_id++) {
    if (contact_flags[leg_id]) { // If foot is in contact

      // Calculate Forward Kinematics from IMU frame to Foot Frame
      Eigen::Matrix4d foot_pose = compute_foot_fk(leg_id, joint_positions);

      // InEKF expects a Kinematics object for correction
      // Note: The covariance matrix (contact noise) is handled via
      // noise_params_ inside the filter
      inekf::Kinematics kin_meas(leg_id, foot_pose, contact_cov);
      kinematics_list.push_back(kin_meas);
      contacts.push_back(std::pair<int, bool>(leg_id, true));
    }
  }
  filter_->setContacts(contacts);

  // Apply right-invariant update
  if (!kinematics_list.empty()) {
    filter_->CorrectKinematics(kinematics_list);
  }

  // 4. Update the exported base_state_
  auto state = filter_->getState();

  Eigen::Vector3d pos = state.getPosition();
  base_state_[0] = pos.x();
  base_state_[1] = pos.y();
  base_state_[2] = pos.z();

  Eigen::Quaterniond quat(state.getRotation());
  base_state_[3] = quat.x();
  base_state_[4] = quat.y();
  base_state_[5] = quat.z();
  base_state_[6] = quat.w(); // Note: Eigen quat constructor is (w,x,y,z), but
                             // outputs components correctly

  Eigen::Vector3d vel = state.getVelocity();
  base_state_[7] = vel.x();
  base_state_[8] = vel.y();
  base_state_[9] = vel.z();

  // Angular velocity is usually taken directly from the bias-corrected gyro
  Eigen::Vector3d gyro_bias = state.getGyroscopeBias();
  Eigen::Vector3d ang_vel = gyro - gyro_bias;
  base_state_[10] = ang_vel.x();
  base_state_[11] = ang_vel.y();
  base_state_[12] = ang_vel.z();

  nav_msgs::msg::Odometry odom = get_odom_msg(pos, quat, vel, ang_vel);

  odom.header.stamp = time;
  odom.header.frame_id = "odom";
  odom.child_frame_id = "base_link";

  odom_pub_->publish(odom);
  return controller_interface::return_type::OK;
}

nav_msgs::msg::Odometry ContactInEKFEstimator::get_odom_msg(
    const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat,
    const Eigen::Vector3d &vel, const Eigen::Vector3d &ang_vel) {
  // 1. Give the time parameter a name in your function signature (e.g., 'const
  // rclcpp::Time & time')

  nav_msgs::msg::Odometry odom;

  // 2. Fill Pose (Position and Orientation in the 'odom' frame)
  odom.pose.pose.position.x = pos.x();
  odom.pose.pose.position.y = pos.y();
  odom.pose.pose.position.z = pos.z();

  odom.pose.pose.orientation.x = quat.x();
  odom.pose.pose.orientation.y = quat.y();
  odom.pose.pose.orientation.z = quat.z();
  odom.pose.pose.orientation.w = quat.w();

  // 3. Fill Twist (Velocities)
  // WARNING: nav_msgs/Odometry expects twist to be in the child_frame_id
  // (base_link). If your filter outputs world-frame velocities, you must rotate
  // them into the base frame. Assuming 'vel' and 'ang_vel' are currently in the
  // world frame:
  Eigen::Vector3d base_vel = quat.inverse() * vel;
  Eigen::Vector3d base_ang_vel = quat.inverse() * ang_vel;

  odom.twist.twist.linear.x = base_vel.x();
  odom.twist.twist.linear.y = base_vel.y();
  odom.twist.twist.linear.z = base_vel.z();

  odom.twist.twist.angular.x = base_ang_vel.x();
  odom.twist.twist.angular.y = base_ang_vel.y();
  odom.twist.twist.angular.z = base_ang_vel.z();

  // 4. Fill Covariances (Optional but highly recommended for Nav2 / SLAM)
  // 36-element array, row-major. Indices 0, 7, 14, 21, 28, 35 are the diagonal.
  // Replace these arbitrary constants with filter_->getCovariance() data if
  // available.
  odom.pose.covariance[0] = 0.01;   // x
  odom.pose.covariance[7] = 0.01;   // y
  odom.pose.covariance[14] = 0.01;  // z
  odom.pose.covariance[21] = 0.001; // roll
  odom.pose.covariance[28] = 0.001; // pitch
  odom.pose.covariance[35] = 0.001; // yaw

  odom.twist.covariance[0] = 0.01;   // vx
  odom.twist.covariance[7] = 0.01;   // vy
  odom.twist.covariance[14] = 0.01;  // vz
  odom.twist.covariance[21] = 0.001; // v_roll
  odom.twist.covariance[28] = 0.001; // v_pitch
  odom.twist.covariance[35] = 0.001; // v_yaw

  return odom;
}

Eigen::Matrix4d ContactInEKFEstimator::compute_foot_fk(
    int leg_id, const std::vector<double> &joint_positions) {

  // Map leg IDs to your URDF foot frame names
  const std::vector<std::string> foot_frame_names = {"FL_foot", "FR_foot",
                                                     "RL_foot", "RR_foot"};

  if (leg_id < 0 || leg_id >= static_cast<int>(foot_frame_names.size())) {
    return Eigen::Matrix4d::Identity();
  }

  // Initialize Pinocchio configuration vector 'q' to neutral stance
  Eigen::VectorXd q = pinocchio::neutral(*pin_model_);

  // Map incoming joint positions (assumes order matches actuated joints in
  // URDF)
  for (size_t i = 0;
       i < joint_positions.size() && (7 + i) < static_cast<size_t>(q.size());
       ++i) {
    q[7 + i] = joint_positions[i];
  }

  // Compute forward kinematics across the kinematic tree
  pinocchio::forwardKinematics(*pin_model_, *pin_data_, q);
  pinocchio::updateFramePlacements(*pin_model_, *pin_data_);

  // Fetch frame IDs for the IMU link and the target foot link
  // (Change "imu_link" to match the exact name of your IMU link in the URDF)
  pinocchio::FrameIndex imu_frame_id = pin_model_->getFrameId("imu_link");
  pinocchio::FrameIndex foot_frame_id =
      pin_model_->getFrameId(foot_frame_names[leg_id]);

  // Extract absolute world placements calculated by Pinocchio
  const pinocchio::SE3 &T_world_imu = pin_data_->oMf[imu_frame_id];
  const pinocchio::SE3 &T_world_foot = pin_data_->oMf[foot_frame_id];

  // Calculate relative SE(3) transform from IMU frame to Foot frame: T_imu_foot
  pinocchio::SE3 T_imu_foot = T_world_imu.inverse() * T_world_foot;

  return T_imu_foot.toHomogeneousMatrix();
}

} // namespace quadruped_estimators

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(quadruped_estimators::ContactInEKFEstimator,
                       controller_interface::ChainableControllerInterface)