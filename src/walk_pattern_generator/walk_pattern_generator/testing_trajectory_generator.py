import rclpy
from rclpy.node import Node
import math
import time

from wbc_control_msgs.msg import WbcCommand, FootTarget
from geometry_msgs.msg import Pose, Twist, Accel, Point, Vector3, Quaternion

class WbcTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('wbc_trajectory_generator')
        
        # Publish at 100Hz
        self.pub = self.create_publisher(WbcCommand, '/wbc_command', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.start_time = time.time()
        
        # Nominal Base Parameters
        self.base_z_nom = 0.30     
        self.orbit_radius = 0.04   
        self.orbit_freq = 0.25     
        
        self.foot_nom_x = [0.12, 0.12, -0.12, -0.12]
        self.foot_nom_y = [0.10, -0.10, 0.10, -0.10]
        
        self.get_logger().info("WBC Trajectory Generator Started.")

    def timer_callback(self):
        msg = WbcCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_heading" 
        
        t = time.time() - self.start_time
        omega = 2.0 * math.pi * self.orbit_freq
        R = self.orbit_radius
        
        msg.base_pose.position.x = float(R * math.sin(omega * t))
        msg.base_pose.position.y = float(R * (1.0 - math.cos(omega * t)))
        msg.base_pose.position.z = float(self.base_z_nom)
        
        msg.base_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        msg.base_velocity.linear.x = float(R * omega * math.cos(omega * t))
        msg.base_velocity.linear.y = float(R * omega * math.sin(omega * t))
        msg.base_velocity.linear.z = 0.0
        
        msg.base_acceleration.linear.x = float(-R * (omega**2) * math.sin(omega * t))
        msg.base_acceleration.linear.y = float(R * (omega**2) * math.cos(omega * t))
        msg.base_acceleration.linear.z = 0.0

        # ---------------------------------------------------------
        # 2. Foot Trajectories (All firmly planted)
        # ---------------------------------------------------------
        for i in range(4):
            msg.foot_targets[i].position.x = float(self.foot_nom_x[i])
            msg.foot_targets[i].position.y = float(self.foot_nom_y[i])
            msg.foot_targets[i].position.z = 0.0
            
            msg.foot_targets[i].velocity = Vector3(x=0.0, y=0.0, z=0.0)
            msg.foot_targets[i].acceleration = Vector3(x=0.0, y=0.0, z=0.0)
            
            # ALL feet remain in stance phase
            msg.foot_targets[i].contact_state = True 

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WbcTrajectoryGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()