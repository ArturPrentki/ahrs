# msg/MpcStateStep.msg

builtin_interfaces/Time time_offset # Time delta/offset relative to trajectory start

# Body SE3 Pose (Translation + Quaternion Orientation)
geometry_msgs/Pose body_pose

# Pinocchio::Motion Spatial Velocity (Linear + Angular)
geometry_msgs/Twist body_velocity

# Pinocchio::Motion Spatial Acceleration (Linear + Angular)
geometry_msgs/Accel body_acceleration

# Foot target states and optimized contact forces
FootState[] feet