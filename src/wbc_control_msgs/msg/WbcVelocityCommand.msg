# WbcVelocityCommand.msg
std_msgs/Header header

# Target body velocity (v_x, v_y, omega_yaw) in heading frame
geometry_msgs/Twist base_velocity

# Target posture
float64 base_height_target   # Desired Z position of the base
float64 base_roll_target     # Desired Roll angle (radians)
float64 base_pitch_target    # Desired Pitch angle (radians)

FootPositionTarget[4] foot_targets