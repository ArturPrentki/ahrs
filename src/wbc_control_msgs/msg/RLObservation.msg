# RLObservation.msg
std_msgs/Header header

# 0: Estimated linear velocity (x, y, z) in base frame (Requires State Estimator)
float32[3] base_lin_vel

# 1: Angular velocity from IMU (x, y, z)
float32[3] base_ang_vel

# 2: Gravity vector projected in base frame (from IMU orientation)
float32[3] projected_gravity

# 3: Target commands (v_x, v_y, omega_yaw) from /vel_cmd
float32[3] velocity_commands

# 4: Joint positions (12 DOF)
float32[12] joint_positions

# 5: Joint velocities (12 DOF)
float32[12] joint_velocities

# 6: Target/Actual local position of the foot relative to base (x, y, z * 4)
float32[12] foot_positions_base_frame

# 7: Contact states (1.0 for stance/contact, 0.0 for swing)
float32[4] foot_contact_states

# 8: The previous action array output by the RL policy
float32[15] last_action