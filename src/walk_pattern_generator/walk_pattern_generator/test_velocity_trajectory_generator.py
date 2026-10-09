import rclpy
from rclpy.node import Node
import math
import time

# Import the new simplified messages
from wbc_control_msgs.msg import WbcVelocityCommand, FootPositionTarget

class WbcVelocityTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('wbc_vel_trajectory_generator')
        
        # Publish at 50Hz (typical RL policy frequency)
        self.pub = self.create_publisher(WbcVelocityCommand, '/wbc_velocity_command', 10)
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        # Give the WBC time to initialize its internal trackers via on_activate()
        time.sleep(1.0) 
        self.start_time = time.time()
        
        # Nominal Parameters
        self.base_z_nom = 0.0    
        self.orbit_radius = 0.06  
        self.orbit_freq = 0.25     
        
        self.foot_nom_x = [0.12, 0.12, -0.12, -0.12]
        self.foot_nom_y = [0.10, -0.10, 0.10, -0.10]
        
        self.get_logger().info("WBC Velocity Trajectory Generator Started.")

    def timer_callback(self):
        msg = WbcVelocityCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_heading" 
        
        t = time.time() - self.start_time
        omega = 2.0 * math.pi * self.orbit_freq
        R = self.orbit_radius
        
        # ---------------------------------------------------------
        # 1. Base Targets (Pure Velocity + Posture)
        # ---------------------------------------------------------
        # Derivative of position: v(t) = R * omega * cos/sin(omega * t)
        msg.base_velocity.linear.x = float(R * omega * math.cos(omega * t))
        msg.base_velocity.linear.y = float(R * omega * math.sin(omega * t))
        msg.base_velocity.linear.z = 0.0
        
        # No rotational velocity; the body stays facing forward
        msg.base_velocity.angular.x = 0.0
        msg.base_velocity.angular.y = 0.0
        msg.base_velocity.angular.z = 0.0
        
        # Absolute posture targets
        msg.base_height_target = float(self.base_z_nom)
        msg.base_roll_target = 0.0
        msg.base_pitch_target = 0.0

        # ---------------------------------------------------------
        # 2. Foot Targets (Only Position + State)
        # ---------------------------------------------------------
        feet = []
        for i in range(4):
            foot = FootPositionTarget()
            foot.position.x = float(self.foot_nom_x[i])
            foot.position.y = float(self.foot_nom_y[i])
            foot.position.z = 0.0
            
            # ALL feet remain in stance phase
            foot.contact = True 
            feet.append(foot)
            
        msg.foot_targets = feet

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WbcVelocityTrajectoryGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()