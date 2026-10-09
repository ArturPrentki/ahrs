# Standard header for timestamping and frame identification
std_msgs/Header header

# --- Base / Centroidal Targets ---
# Target body pose (Position + Orientation via Quaternion)
geometry_msgs/Pose base_pose

# Target body velocity (Linear + Angular)
geometry_msgs/Twist base_velocity

# Target body acceleration (Linear + Angular)
geometry_msgs/Accel base_acceleration

# --- Foot Targets ---
# Array of 4 feet. 
# Convention MUST be maintained (e.g., 0: Front-Left, 1: Front-Right, 2: Rear-Left, 3: Rear-Right)
FootTarget[4] foot_targets