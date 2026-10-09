#pragma once
#include "whole_body_controller/system_state.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <OsqpEigen/OsqpEigen.h>

namespace wbc {
class Optimizer {
public:
  Optimizer();
  ~Optimizer() = default;

  // initialize using dynamic system as reference for matrices
  bool init();

  bool solve(const SystemState &current, const TargetState &target,
             ControlOutput &output);

private:
  static constexpr int base_dofs = 6;
  static constexpr int nq = 19;
  static constexpr int n_forces = 12;
  static constexpr int num_variables = SystemConsts::nq_dot + n_forces;
  static constexpr int num_constraints = 38;
  void build_cost(const SystemState &current_state,
                  const TargetState &target_state);
  void build_dynamics_constraints(
      const Eigen::Vector<double, base_dofs> &hu,
      const Eigen::MatrixXd &mass_matrix,
      const Eigen::Matrix<double, 12, SystemConsts::nq_dot> &contact_jacobian);
  void build_contact_constraints(
      const SystemState &system_state,
      const TargetState &target_state);
  void build_friction_constraints(const TargetState &target_state);

  OsqpEigen::Solver solver_;
  // Pre-allocated Matrices
  std::vector<int> leg_v_idx_;
  Eigen::Matrix<double, num_variables, num_variables> dense_hesian_;
  Eigen::Matrix<double, num_constraints, num_variables>
      dense_linear_constraints_;
  Eigen::SparseMatrix<double> hessian_;
  Eigen::SparseMatrix<double> linear_constraints_;
  // Eigen::Matrix<double, 12, nq_dot> J_contact_;
  // Eigen::Matrix<double, 6, nq_dot> J_base;
  Eigen::Matrix<double, 12, SystemConsts::nq_dot> S_joint;
  Eigen::Matrix<double, num_variables, 1> gradient_;
  Eigen::Matrix<double, num_constraints, 1> lower_bound_, upper_bound_;

  Eigen::Matrix<double, 6, 6> optimization_weight_base_;
  Eigen::Matrix<double, 12, 12> optimization_weight_joint_;
  Eigen::Matrix<double, n_forces, n_forces> optimization_weight_force_;
  Eigen::Matrix3d optimization_weight_foot_;

  Eigen::Matrix3d Kp_foot_;
  Eigen::Matrix3d Kd_foot_;

  double Kp_base_lin_, Kd_base_lin_;
  double Kp_base_ang_, Kd_base_ang_;
};
} // namespace wbc