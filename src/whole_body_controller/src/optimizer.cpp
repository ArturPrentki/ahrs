#include "whole_body_controller/optimizer.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "whole_body_controller/system_state.hpp"
#include <Eigen/Dense>
#include <array>

namespace wbc {
template <typename SparseMatrixType, typename DenseMatrixType>
static void update_sparse_matrix(SparseMatrixType &sparse_mat,
                                 const DenseMatrixType &dense_mat) {
  for (int k = 0; k < sparse_mat.outerSize(); ++k) {
    for (typename SparseMatrixType::InnerIterator it(sparse_mat, k); it; ++it) {
      it.valueRef() = dense_mat(it.row(), it.col());
    }
  }
}

Optimizer::Optimizer() {}
bool Optimizer::init() {
  dense_hesian_.setZero();
  dense_linear_constraints_.setZero();

  S_joint.setZero();
  gradient_.setZero();
  lower_bound_.setZero();
  upper_bound_.setZero();
  Eigen::Matrix<double, num_variables, num_variables> H_template;
  H_template.setZero();

  H_template.topLeftCorner(num_variables, num_variables)
      .triangularView<Eigen::Upper>()
      .setOnes();
  H_template.bottomRightCorner(n_forces, n_forces).setIdentity();

  // Lock structure! (Since it's filled with 1s, no elements are dropped)
  hessian_ = H_template.sparseView();
  Eigen::Matrix<double, num_constraints, num_variables>
      linear_constraints_template;
  linear_constraints_template.setZero();

  linear_constraints_template.block(0, 0, 6, num_variables).setOnes();
  linear_constraints_template.block(6, 0, 12, num_variables).setOnes();

  linear_constraints_template.block(18, SystemConsts::nq_dot, 20, n_forces)
      .setOnes();

  // Lock structure!
  linear_constraints_ = linear_constraints_template.sparseView();

  hessian_.makeCompressed();
  linear_constraints_.makeCompressed();

  // --- 2. Configure Solver Settings ---
  solver_.settings()->setVerbosity(false);
  solver_.settings()->setWarmStart(true);
  solver_.settings()->setAdaptiveRho(true);

  // --- 3. Pass Data to OSQP Data Workspace ---
  solver_.data()->setNumberOfVariables(num_variables);
  solver_.data()->setNumberOfConstraints(num_constraints);

  if (!solver_.data()->setHessianMatrix(hessian_))
    return false;
  if (!solver_.data()->setGradient(gradient_))
    return false;
  if (!solver_.data()->setLinearConstraintsMatrix(linear_constraints_))
    return false;
  if (!solver_.data()->setLowerBound(lower_bound_))
    return false;
  if (!solver_.data()->setUpperBound(upper_bound_))
    return false;

  Kp_foot_ = Eigen::Matrix3d::Identity() * 100.0;
  Kd_foot_ = Eigen::Matrix3d::Identity() * 15.0;
  optimization_weight_foot_ = Eigen::Matrix3d::Identity() * 40.0;

  Eigen::Matrix<double, 6, 1> base_weights;
  // clang-format off
  base_weights << 
      30.0, 30.0, 60.0, // linear x,y,z
      60.0, 50.0, 60.0; // angular x,y,z
  // clang-format on
  Kp_base_lin_ = 5.;
  Kd_base_lin_ = .8;
  Kp_base_ang_ = 5.;
  Kd_base_ang_ = .8;

  optimization_weight_base_ = base_weights.asDiagonal();

  optimization_weight_joint_ =
      decltype(optimization_weight_joint_)::Identity() * 1e-4;

  optimization_weight_force_ =
      2. * 1e-3 * Eigen::Matrix<double, n_forces, n_forces>::Identity();
  return solver_.initSolver();
};

bool Optimizer::solve(const SystemState &system_state,
                      const TargetState &target_state, ControlOutput &output) {
  dense_hesian_.setZero();
  dense_linear_constraints_.setZero();

  build_cost(system_state, target_state);
  build_dynamics_constraints(system_state.hu, system_state.mass_matrix,
                             system_state.foot_jacobian);

  build_contact_constraints(system_state, target_state);

  build_friction_constraints(target_state);

  update_sparse_matrix(hessian_, dense_hesian_);
  update_sparse_matrix(linear_constraints_, dense_linear_constraints_);

  solver_.updateHessianMatrix(hessian_);
  solver_.updateGradient(gradient_);
  solver_.updateLinearConstraintsMatrix(linear_constraints_);
  solver_.updateBounds(lower_bound_, upper_bound_);

  if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
    return false;
  }

  const auto &solution = solver_.getSolution();
  output.q_ddot = solution.head(SystemConsts::nq_dot);
  output.contact_forces = solution.tail(n_forces);

  return true;
}
void Optimizer::build_cost(const SystemState &system_state,
                           const TargetState &target) {

  // =====================================================================
  // 1. BASE TASK (SE3 Manifold tracking)
  // =====================================================================
  Eigen::Quaterniond quat_curr(system_state.q[6], system_state.q[3], system_state.q[4],
                               system_state.q[5]);
  Eigen::Quaterniond quat_targ(target.q[6], target.q[3], target.q[4],
                               target.q[5]);

  pinocchio::SE3 pose_curr(quat_curr, system_state.q.head<3>());
  pinocchio::SE3 pose_targ(quat_targ, target.q.head<3>());

  pinocchio::Motion twist_err_local =
      pinocchio::log6(pose_curr.actInv(pose_targ));

  Eigen::Matrix<double, 6, 1> spatial_error;
  spatial_error << pose_curr.rotation() * twist_err_local.linear(),
      pose_curr.rotation() * twist_err_local.angular();

  Eigen::Matrix<double, 6, 1> vel_error =
      target.q_dot.head<6>() - system_state.q_dot.head<6>();

  Eigen::Matrix<double, 6, 1> base_pd;
  base_pd.head<3>() = Kp_base_lin_ * spatial_error.head<3>() +
                      Kd_base_lin_ * vel_error.head<3>();
  base_pd.tail<3>() = Kp_base_ang_ * spatial_error.tail<3>() +
                      Kd_base_ang_ * vel_error.tail<3>();

  Eigen::Matrix<double, 6, 1> rhs_base =
      target.q_ddot.head<6>() + base_pd - system_state.base_drift;

  // =====================================================================
  // 2. JOINT POSTURAL TASK (Vectorized PD control)
  // =====================================================================
  Eigen::Vector<double, 12> j_acc =
      target.q_ddot.tail<12>() +
      target.j_kp.cwiseProduct(target.q.tail<12>() - system_state.q.tail<12>()) -
      target.j_kd.cwiseProduct(target.q_dot.tail<12>() -
                               system_state.q_dot.tail<12>());

  // =====================================================================
  // 3. POPULATE HESSIAN & GRADIENT (Block operations, no S_joint)
  // =====================================================================
  // Base task contribution
  dense_hesian_.topLeftCorner<SystemConsts::nq_dot, SystemConsts::nq_dot>() =
      2.0 * system_state.base_jacobian.transpose() * optimization_weight_base_ *
      system_state.base_jacobian;

  gradient_.head<SystemConsts::nq_dot>() = -2.0 *
                                           system_state.base_jacobian.transpose() *
                                           optimization_weight_base_ * rhs_base;

  // Joint task contribution (written directly into the 12x12 bottom-right of
  // the q_dot block)
  const double acc_regularization = 1e-4;
  dense_hesian_.topLeftCorner<SystemConsts::nq_dot, SystemConsts::nq_dot>()
      .diagonal()
      .array() += acc_regularization;

  dense_hesian_.block<12, 12>(6, 6) += 2.0 * optimization_weight_joint_;
  gradient_.segment<12>(6) -= 2.0 * optimization_weight_joint_ * j_acc;

  // Force regularization contribution
  dense_hesian_.bottomRightCorner(n_forces, n_forces) =
      2.0 * optimization_weight_force_;

  gradient_.tail(n_forces) =
      -2.0 * optimization_weight_force_ * target.contact_forces;

  // =====================================================================
  // 4. SWING FOOT TASKS
  // =====================================================================
  for (int i = 0; i < 4; ++i) {
    if (!target.fs[i].contact) {
      Eigen::Matrix<double, 3, SystemConsts::nq_dot> J_f =
          system_state.foot_jacobian.middleRows<3>(3 * i);

      Eigen::Vector3d rhs_foot =
          Kp_foot_ * (target.fs[i].foot_pos - system_state.fs[i].foot_pos) +
          Kd_foot_ * (target.fs[i].foot_vel - system_state.fs[i].foot_vel) -
          system_state.foot_drift[i]; // (Assuming target foot acceleration is 0)

      dense_hesian_
          .topLeftCorner<SystemConsts::nq_dot, SystemConsts::nq_dot>() +=
          2.0 * J_f.transpose() * optimization_weight_foot_ * J_f;

      gradient_.head<SystemConsts::nq_dot>() -=
          2.0 * J_f.transpose() * optimization_weight_foot_ * rhs_foot;
    }
  }
}

void Optimizer::build_dynamics_constraints(
    const Eigen::Vector<double, base_dofs> &hu,
    const Eigen::MatrixXd &mass_matrix,
    const Eigen::Matrix<double, 12, SystemConsts::nq_dot> &contact_jacobian) {
  dense_linear_constraints_.block(0, 0, base_dofs, SystemConsts::nq_dot) =
      mass_matrix.topRows(base_dofs);

  Eigen::Matrix<double, 6, 12> J_c_top_trans =
      contact_jacobian.transpose().topRows(base_dofs); // 6 x 12

  dense_linear_constraints_.block(0, SystemConsts::nq_dot, base_dofs, 12) =
      -J_c_top_trans;

  // Set Bounds: M_u * qdd - J_c^T * F = -h_u
  lower_bound_.segment(0, base_dofs) = -hu;
  upper_bound_.segment(0, base_dofs) = -hu;
}

void Optimizer::build_contact_constraints(const SystemState &system_state,
                                          const TargetState &target) {
  // Fill Contact Jacobians [ J_c   0 ]
  dense_linear_constraints_.block(6, 0, 12, SystemConsts::nq_dot) =
      system_state.foot_jacobian;

  for (int i = 0; i < 4; ++i) {
    if (target.fs[i].contact) {
      // J_c * qdd = -J_dot * q_dot

      Eigen::Vector3d p_err =
          target.fs[i].foot_pos - system_state.fs[i].foot_pos;
      Eigen::Vector3d v_err =
          target.fs[i].foot_vel - system_state.fs[i].foot_vel;

      // Small gains are usually enough (e.g., Kp = 50, Kd = 5)
      

      Eigen::Vector3d foot_correction = Kp_foot_ * p_err + Kd_foot_ * v_err;

      // Update BOTH bounds
      lower_bound_.segment(6 + 3 * i, 3) =
          -system_state.foot_drift[i] + foot_correction;
      upper_bound_.segment(6 + 3 * i, 3) =
          -system_state.foot_drift[i] + foot_correction;
    } else { // Swing foot
      lower_bound_.segment(6 + 3 * i, 3).setConstant(-OsqpEigen::INFTY);
      upper_bound_.segment(6 + 3 * i, 3).setConstant(OsqpEigen::INFTY);
    }
  }
}
void Optimizer::build_friction_constraints(const TargetState &target) {

  const double mu = 1.5;       // Friction coefficient
  const double Fz_max = 500.0; // Max normal force per leg

  // Matrix block for linearized friction pyramid
  Eigen::Matrix<double, 4, 3> friction_pyramid;
  friction_pyramid << 1, 0, -mu, -1, 0, -mu, 0, 1, -mu, 0, -1, -mu;

  for (int i = 0; i < 4; ++i) {
    int f_idx =
        SystemConsts::nq_dot + 3 * i; // Index where this foot's forces start

    dense_linear_constraints_.block(18 + 4 * i, f_idx, 4, 3) = friction_pyramid;
    dense_linear_constraints_(34 + i, f_idx + 2) = 1.0; // Select Fz
    if (target.fs[i].contact) {
      // contact force(Z) is larger than 1.0 newton
      lower_bound_.segment(18 + 4 * i, 4).setConstant(-OsqpEigen::INFTY);
      upper_bound_.segment(18 + 4 * i, 4).setConstant(0.0);
      lower_bound_(34 + i) = 10.0; // Fz_min: Must push down
      upper_bound_(34 + i) = Fz_max;
    } else {
      lower_bound_.segment(18 + 4 * i, 4).setConstant(0.0);
      upper_bound_.segment(18 + 4 * i, 4).setConstant(0.0);
      lower_bound_(34 + i) = 0.0;
      upper_bound_(34 + i) = 0.0;
    }
  }
}

} // namespace wbc