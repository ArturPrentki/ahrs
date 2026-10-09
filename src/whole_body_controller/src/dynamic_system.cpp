#include "whole_body_controller/dynamic_system.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/spatial/fwd.hpp"
#include "whole_body_controller/system_state.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <vector>

namespace wbc {

bool DynamicSystem::initialize(const std::string &urdf_string,
                               const std::string &base_name,
                               const std::vector<std::string> &hip_names,
                               const std::vector<std::string> &thigh_names,
                               const std::vector<std::string> &calf_names,
                               const std::vector<std::string> &feet_names,
                               const std::vector<std::string> &joints_names) {
  pinocchio::urdf::buildModelFromXML(
      urdf_string, pinocchio::JointModelFreeFlyer(), pin_model_);
  pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
  kin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

  if (!pin_model_.existFrame(base_name)) {
    return false;
  }

  if (feet_names.size() != 4 || hip_names.size() != 4) {
    return false; // Prevent hardcoded out-of-bounds access later
  }
  base_frame_id_ = pin_model_.getFrameId(base_name);

  for (int i = 0; i < 4; ++i) {
    hip_ids_[i] = pin_model_.getJointId(hip_names[i]);
    thigh_ids_[i] = pin_model_.getJointId(thigh_names[i]);
    calf_ids_[i] = pin_model_.getJointId(calf_names[i]);
    foot_ids_[i] = pin_model_.getFrameId(feet_names[i]);
  }

  for (int i = 0; i < 12; ++i) {

    if (!pin_model_.existJointName(joints_names[i]))
      return false;
    joint_v_indices_[i] =
        pin_model_.joints[pin_model_.getJointId(joints_names[i])].idx_v();
  }

  return true;
}

void DynamicSystem::inverse_kinematics(TargetState &target) const {
  // 1. Reconstruct floating-base state
  Eigen::Vector3d body_pos(target.q[0], target.q[1], target.q[2]);
  Eigen::Quaterniond body_q(target.q[6], target.q[3], target.q[4],
                            target.q[5]); // (w, x, y, z)

  Eigen::Vector3d v_body(target.q_dot[0], target.q_dot[1], target.q_dot[2]);
  Eigen::Vector3d w_body(target.q_dot[3], target.q_dot[4], target.q_dot[5]);

  pinocchio::SE3 odom_to_base(body_q.toRotationMatrix(), body_pos);

  for (int i = 0; i < 4; ++i) {
    // --- 2. TRANSFORMS & POSITION IK ---
    const pinocchio::SE3 &base_to_hip = pin_model_.jointPlacements[hip_ids_[i]];
    pinocchio::SE3 odom_to_hip = odom_to_base * base_to_hip;

    // Local foot position in hip frame
    Eigen::Vector3d foot_in_hip = odom_to_hip.actInv(target.fs[i].foot_pos);

    double x = foot_in_hip.x();
    double y = foot_in_hip.y();
    double z = foot_in_hip.z();

    // Link lengths from Pinocchio joint placements
    const pinocchio::SE3 &hip_to_thigh =
        pin_model_.jointPlacements[thigh_ids_[i]];
    double mul = (hip_to_thigh.translation().y() > 0) ? 1.0 : -1.0;
    double l_hip = hip_to_thigh.translation().norm() * mul;
    const pinocchio::SE3 &thigh_to_calf =
        pin_model_.jointPlacements[calf_ids_[i]];
    double l_thigh = thigh_to_calf.translation().norm();
    const pinocchio::SE3 &calf_to_foot =
        pin_model_.frames[foot_ids_[i]].placement;
    double l_calf = calf_to_foot.translation().norm();

    // Position IK
    double L2 = x * x + y * y + z * z - l_hip * l_hip;
    double L = std::sqrt(std::max(0.0, L2));

    double q1 = std::atan2(y, -z) - std::atan2(l_hip, L);
    double cos_q3 =
        (L2 - l_thigh * l_thigh - l_calf * l_calf) / (2.0 * l_thigh * l_calf);
    double q3 = -std::acos(std::clamp(cos_q3, -1.0, 1.0));
    double q2 = std::atan2(-x, L) - std::atan2(l_calf * std::sin(q3),
                                               l_thigh + l_calf * std::cos(q3));

    // Save joint angles
    target.q.segment<3>(7 + i * 3) << q1, q2, q3;

    // --- 3. VELOCITY IK (q_dot) ---

    // Foot linear velocity relative to hip origin (in World frame)
    Eigen::Vector3d r_foot_world = target.fs[i].foot_pos - body_pos;
    Eigen::Vector3d v_foot_rel_world =
        target.fs[i].foot_vel - v_body - w_body.cross(r_foot_world);

    // Rotate foot velocity into Hip local frame
    Eigen::Vector3d v_hip =
        odom_to_hip.rotation().transpose() * v_foot_rel_world;

    // Construct 3x3 Analytical Leg Jacobian J_leg in Hip Frame: v_hip = J_leg *
    // qdot
    double s1 = std::sin(q1), c1 = std::cos(q1);
    double s2 = std::sin(q2), c2 = std::cos(q2);
    double s23 = std::sin(q2 + q3), c23 = std::cos(q2 + q3);

    Eigen::Matrix3d J_leg;

    // Column 1: d(foot)/dq1
    J_leg(0, 0) = 0.0;
    J_leg(1, 0) = -l_hip * s1 - c1 * (l_thigh * c2 + l_calf * c23);
    J_leg(2, 0) = -l_hip * c1 + s1 * (l_thigh * c2 + l_calf * c23);

    // Column 2: d(foot)/dq2
    J_leg(0, 1) = -l_thigh * c2 - l_calf * c23;
    J_leg(1, 1) = -s1 * (l_thigh * s2 + l_calf * s23);
    J_leg(2, 1) = c1 * (l_thigh * s2 + l_calf * s23);

    // Column 3: d(foot)/dq3
    J_leg(0, 2) = -l_calf * c23;
    J_leg(1, 2) = -l_calf * s1 * s23;
    J_leg(2, 2) = l_calf * c1 * s23;

    // Solve q_dot = J_leg^-1 * v_hip
    // ColPivHouseholderQR handles potential singularities gracefully
    Eigen::Vector3d qdot = J_leg.colPivHouseholderQr().solve(v_hip);

    // Save joint velocities to state.qd (assuming offset 6 for base velocities
    // [v, w])
    target.q_dot.segment<3>(6 + i * 3) = qdot;
  }
}

void DynamicSystem::inverse_dynamics(const SystemState &system_state,
                                     ControlOutput &target) const {

  Eigen::Matrix<double, 18, 1> tau_full =
      pin_data_->M * target.q_ddot + pin_data_->nle -
      system_state.foot_jacobian.transpose() * target.contact_forces;

  Eigen::Vector<double, 12> joint_torques;
  for (int i = 0; i < 12; ++i) {
    joint_torques[i] = tau_full[joint_v_indices_[i]];
  }

  target.torque = joint_torques;
}
void DynamicSystem::update_dynamics(SystemState &system_state) {
  pinocchio::computeAllTerms(pin_model_, *pin_data_, system_state.q,
                             system_state.q_dot);

  // pinocchio::forwardKinematics(pin_model_, *pin_data_, q_, q_dot_,
  //                              q_ddot_zero_);
  pinocchio::updateFramePlacements(pin_model_, *pin_data_);
  pinocchio::forwardKinematics(pin_model_, *pin_data_, system_state.q,
                               system_state.q_dot, q_ddot_zero_);
  pinocchio::getFrameJacobian(pin_model_, *pin_data_, base_frame_id_,
                              pinocchio::LOCAL_WORLD_ALIGNED,
                              system_state.base_jacobian);

  pin_data_->M.triangularView<Eigen::StrictlyLower>() =
      pin_data_->M.transpose().triangularView<Eigen::StrictlyLower>();

  // formulate jacobians
  // populate foot acceleration biases
  system_state.base_drift = pinocchio::getFrameClassicalAcceleration(
      pin_model_, *pin_data_, base_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
  Eigen::Matrix<double, 6, SystemConsts::nq_dot> J_temp_foot;

  system_state.mass_matrix = pin_data_->M;
  system_state.hu = pin_data_->nle.head<SystemConsts::base_dof>();
  for (int i = 0; i < 4; ++i) {
    J_temp_foot.setZero();
    pinocchio::getFrameJacobian(pin_model_, *pin_data_, foot_ids_[i],
                                pinocchio::LOCAL_WORLD_ALIGNED, J_temp_foot);

    system_state.foot_jacobian.block<3, SystemConsts::nq_dot>(3 * i, 0) =
        J_temp_foot.topRows(3);
    system_state.foot_drift[i] = pinocchio::getFrameClassicalAcceleration(
                                     pin_model_, *pin_data_, foot_ids_[i],
                                     pinocchio::LOCAL_WORLD_ALIGNED)
                                     .linear();

    system_state.fs[i].foot_pos = pin_data_->oMf[foot_ids_[i]].translation();
    system_state.fs[i].foot_vel =
        pinocchio::getFrameVelocity(pin_model_, *pin_data_, foot_ids_[i],
                                    pinocchio::LOCAL_WORLD_ALIGNED)
            .linear();

    Eigen::Matrix3d J_leg_linear;
    Eigen::Vector3d tau_leg;

    for (int j = 0; j < 3; ++j) {
      // The index of the joint in your custom arrays (0 to 11)
      int joint_idx = i * 3 + j;

      // The column in Pinocchio's Jacobian for this specific joint
      int pin_col = joint_v_indices_[joint_idx];

      // Slot the specific column and torque together
      J_leg_linear.col(j) =
          system_state.foot_jacobian.block<3, 1>(3 * i, pin_col);
      tau_leg(j) = system_state.j_torq(joint_idx);
    }

    Eigen::Vector3d force = -J_leg_linear.transpose().inverse() * tau_leg;
    // std::cout << "Force " << joint_v_indices_[i * 3] << " GRF " << force.transpose() <<" T "<<tau_leg.transpose()
    //           << std::endl;
    // std::cout<< "joints: " << i << " " <<force.transpose() << std::endl;

    system_state.fs[i].contact = (force.z() > 9.81);
  }
}
} // namespace wbc