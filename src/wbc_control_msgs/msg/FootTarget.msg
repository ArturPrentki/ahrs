# Target position of the foot (x, y, z)
geometry_msgs/Point position

# Target linear velocity of the foot (vx, vy, vz)
geometry_msgs/Vector3 velocity

# Target linear acceleration of the foot (ax, ay, az)
geometry_msgs/Vector3 acceleration

# Desired contact state: true (1) for stance/contact, false (0) for swing
bool contact_state