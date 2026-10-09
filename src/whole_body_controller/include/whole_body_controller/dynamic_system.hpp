#pragma once
#include "Eigen/Dense"

#include <Eigen/src/Core/Matrix.h>
#include <string>
#include <vector>

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody.hpp>

#include "pinocchio/multibody/fwd.hpp"
#include "system_state.hpp"
#include <array>
#include <pinocchio/parsers/urdf.hpp>

namespace wbc {

class DynamicSystem {
  // pinocchio handling class
private:
  // prealocated state representation for pinocchios internal use

  pinocchio::Model pin_model_;
  std::unique_ptr<pinocchio::Data> pin_data_;
  std::unique_ptr<pinocchio::Data> kin_data_;
  int base_frame_id_;
  std::array<int, 4> foot_ids_;
  std::array<int, 4> hip_ids_;
  std::array<int, 4> thigh_ids_;
  std::array<int, 4> calf_ids_;

  std::array<int, 12> joint_v_indices_;

  std::array<Eigen::Vector3d, 4> feet_drift_;
  Eigen::Vector<double, 6> base_drift_;

  const Eigen::Vector<double, SystemConsts::nq_dot> q_ddot_zero_ =
      Eigen::Vector<double, SystemConsts::nq_dot>::Zero();

public:
  const double &get_total_mass() const { return pin_data_->mass[0]; }
  const pinocchio::Model &get_model() const { return pin_model_; }
  bool initialize(const std::string &urdf_string, const std::string &base_name,
                  const std::vector<std::string> &hip_names,
                  const std::vector<std::string> &thigh_names,
                  const std::vector<std::string> &calf_names,
                  const std::vector<std::string> &feet_names,
                  const std::vector<std::string> &joints_names);

  void inverse_kinematics(TargetState &state) const;

  void update_dynamics(SystemState &state);

  void inverse_dynamics(const SystemState &state, ControlOutput &target) const;


};
} // namespace wbc