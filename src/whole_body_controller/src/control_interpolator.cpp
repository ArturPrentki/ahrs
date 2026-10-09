#include "whole_body_controller/control_interpolator.hpp"
#include "whole_body_controller/system_state.hpp"
#include <cstddef>
#include <pinocchio/spatial/explog.hpp>

namespace wbc {

TargetState &ControlInterpolator::get_next_target() {
  std::vector<TargetState> *mpc_trajectory = targets_.readFromRT();
  
  if (!mpc_trajectory || mpc_trajectory->empty()) {
    current_target_ = default_state_;
    current_target_.q_dot.setZero();
    current_target_.q_ddot.setZero();
    return current_target_;
  }

  // Reset tick if new MPC data arrived
  if (new_data_flag_.exchange(false, std::memory_order_acquire)) {
    tick_ = 0;
  }
  
  size_t max_idx = mpc_trajectory->size() - 1;
  size_t id_inter_a = std::min(static_cast<size_t>(tick_ / config_.ticks_per_sample), max_idx);
  size_t id_inter_b = std::min(id_inter_a + 1, max_idx);

  if (tick_ < config_.horizon_ticks) {
    // --- 1. NORMAL INTERPOLATION PHASE ---
    
    TargetState inter_a_ref = (id_inter_a == 0) ? default_state_ : (*mpc_trajectory)[id_inter_a];
    TargetState inter_b_ref = (*mpc_trajectory)[id_inter_b];

    int tick_in_this_inter = tick_ - id_inter_a * config_.ticks_per_sample;
    double alpha = static_cast<double>(tick_in_this_inter) / config_.ticks_per_sample;
    
    // Clamp alpha to 1.0 just in case id_inter_a hits max_idx early
    alpha = std::min(alpha, 1.0);

    pinocchio::interpolate(model_, inter_a_ref.q, inter_b_ref.q, alpha, current_target_.q);
    current_target_.q_dot = inter_a_ref.q_dot * (1.0 - alpha) + inter_b_ref.q_dot * alpha;
    current_target_.q_ddot = inter_a_ref.q_ddot * (1.0 - alpha) + inter_b_ref.q_ddot * alpha;

    for(size_t i = 0; i < 12; ++i) {
      current_target_.contact_forces[i] = inter_a_ref.contact_forces[i] * (1.0 - alpha) + inter_b_ref.contact_forces[i] * alpha;
    }

    for (size_t i = 0; i < 4; ++i) {
      current_target_.fs[i].contact = inter_b_ref.fs[i].contact; // Step function for contact state
      current_target_.fs[i].foot_pos = inter_a_ref.fs[i].foot_pos * (1.0 - alpha) + inter_b_ref.fs[i].foot_pos * alpha;
      current_target_.fs[i].foot_vel = inter_a_ref.fs[i].foot_vel * (1.0 - alpha) + inter_b_ref.fs[i].foot_vel * alpha;
    }

  } else if (tick_ < config_.horizon_ticks + config_.extrapolation_ticks) {
    // --- 2. EXTRAPOLATION PHASE ---
    
    // Notice we DO NOT reset current_target_ to default_state_. 
    // It seamlessly retains the position/forces from the last valid interpolation tick.
    int extra_ticks = tick_ - config_.horizon_ticks;
    double decay = 1.0 - (static_cast<double>(extra_ticks) / config_.extrapolation_ticks);

    // Decay dynamics to zero smoothly
    current_target_.q_dot *= decay;
    current_target_.q_ddot *= decay;
    
    // Optional but recommended: Decay contact forces to zero to relax the robot
    for(size_t i = 0; i < 12; ++i) {
        current_target_.contact_forces[i] *= decay;
    }
    
  } else {
    // --- 3. RECOVERY PHASE ---
    // At this point, MPC is completely dead. 
    // Velocities are zero. Keep configuration static.
    current_target_.q_dot.setZero();
    current_target_.q_ddot.setZero();
  }

  tick_++;
  return current_target_;
}
} // namespace wbc
