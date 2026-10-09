# RLActionCommand.msg
std_msgs/Header header

# 1. Posture Targets (3 dimensions)
float32 base_height_target
float32 base_roll_target
float32 base_pitch_target

# 2. Foot Position Targets (12 dimensions)
# Order MUST match the URDF simulation index: 
# 0: FL (Front Left)
# 1: FR (Front Right)
# 2: RL (Rear Left)
# 3: RR (Rear Right)
FootPositionTarget[4] foot_targets