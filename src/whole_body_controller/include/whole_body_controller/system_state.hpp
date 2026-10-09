#pragma once
#include "Eigen/Dense"
#include "array"
#include <Eigen/src/Core/Matrix.h>

namespace wbc {
struct SystemConsts {
  static constexpr int nq = 19;
  static constexpr int nq_dot = 18;
  static constexpr int base_dof = 6;
  static constexpr int actuated_dof = 12;
};

struct FootState {
  bool contact; // actual contact tracking
  Eigen::Vector3d foot_pos;
  Eigen::Vector3d foot_vel;
};
inline std::ostream &operator<<(std::ostream &os, const FootState &fs) {
  os << "{ contact: " << (fs.contact ? "true" : "false") << ", pos: ["
     << fs.foot_pos.transpose() << "]";
  return os;
}

struct SystemState {

  Eigen::Vector<double, 19> q;
  Eigen::Vector<double, 18> q_dot;
  Eigen::Vector<double, 12> j_torq;
  std::array<FootState, 4> fs;

  Eigen::Matrix<double, SystemConsts::base_dof, SystemConsts::nq_dot>
      base_jacobian;
  Eigen::Matrix<double, 12, SystemConsts::nq_dot> foot_jacobian;
  std::array<Eigen::Vector3d, 4> foot_drift;
  Eigen::Vector<double, 6> base_drift;
  Eigen::Matrix<double, SystemConsts::base_dof, 1> hu;
  Eigen::MatrixXd mass_matrix;
};

inline std::ostream &operator<<(std::ostream &os, const SystemState &ss) {
  os << "SystemState {\n"
     << "Base: {\n"
     << "  q: [" << ss.q.head<7>().transpose() << "]\n"
     << "  v: [" << ss.q_dot.head<6>().transpose() << "]\n"
     << "}\nJoints: {\n"
     << "  q: [" << ss.q.tail<12>().transpose() << "]\n"
     << "  v: [" << ss.q_dot.tail<12>().transpose() << "]\n}\n"
     << "Feet: [\n";
  for (size_t i = 0; i < ss.fs.size(); ++i) {
    os << ss.fs[i] << "\n"; //(i + 1 < ss.fs.size() ? ", " : "");
  }
  os << " ]\n";
  return os;
}

struct TargetState {
  Eigen::Vector<double, 19> q;
  Eigen::Vector<double, 18> q_dot;
  Eigen::Vector<double, 18> q_ddot;
  Eigen::Vector<double, 12> contact_forces;

  std::array<FootState, 4> fs;
  Eigen::Vector<double, 12> j_kp;
  Eigen::Vector<double, 12> j_kd;
};
inline std::ostream &operator<<(std::ostream &os, const TargetState &ts) {
  os << "TargetState {\n"
     << "  q: [" << ts.q.transpose() << "]\n"
     << "  q_dot: [" << ts.q_dot.transpose() << "]\n"
     << "  q_ddot: [" << ts.q_ddot.transpose() << "]\n"

     << "  fs: [ ";
  for (size_t i = 0; i < ts.fs.size(); ++i) {
    os << ts.fs[i] << (i + 1 < ts.fs.size() ? ", " : "");
  }
  os << " ]\n"
     << "  contact_forces: [" << ts.contact_forces.transpose() << "]\n"
     << "}";
  return os;
}
struct ControlOutput {
  Eigen::Vector<double, 18> q_ddot;
  Eigen::Vector<double, 12> contact_forces;
  Eigen::Vector<double, 12> torque;
};
inline std::ostream &operator<<(std::ostream &os, const ControlOutput &co) {
  os << "ControlOutput {\n"
     << "  q_ddot: [" << co.q_ddot.transpose() << "]\n"
     << "  contact_forces: [" << co.contact_forces.transpose() << "]\n"
     << "  torque: [" << co.torque.transpose() << "]\n"
     << "}";
  return os;
}
} // namespace wbc