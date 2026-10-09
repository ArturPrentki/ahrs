string name                         # Foot identifier ("FL", "FR", "RL", "RR")
bool is_contact                     # Contact state (true = stance, false = swing)
geometry_msgs/Point position        # Foot target position (World Frame) ????
geometry_msgs/Vector3 velocity       # Foot target velocity
geometry_msgs/Vector3 contact_force # Optimized ground reaction force (Fx, Fy, Fz)