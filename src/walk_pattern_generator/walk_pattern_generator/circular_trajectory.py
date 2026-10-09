import rclpy
from rclpy.node import Node
import math
import time

from wbc_control_msgs.msg import MpcTrajectory, MpcStateStep, FootState 

class CircularTrajectoryNode(Node):
    def __init__(self):
        super().__init__('circular_trajectory_node')
        
        # Publisher
        self.traj_pub = self.create_publisher(MpcTrajectory, '/mpc_trajectory', 10)
        
        # Trajectory parameters
        self.radius = 0.04       # 4 cm circle radius (safe starting value)
        self.freq = 0.25          # 0.5 Hz (2 seconds per revolution)
        self.z_height = 0.3     # Nominal standing height
        
        # MPC Horizon parameters
        self.horizon_length = 100 # Number of steps to send ahead 1s
        self.dt = 0.01          # 10ms per MPC step (100Hz)
        
        # Order: FR, FL, RR, RL
        self.nom_feet = [
            {'pos': [ 0.19, -0.142, 0.0], 'force': [0.0, 0.0, 20.0]}, # FR
            {'pos': [ 0.19,  0.142, 0.0], 'force': [0.0, 0.0, 20.0]}, # FL
            {'pos': [-0.19, -0.142, 0.0], 'force': [0.0, 0.0, 20.0]}, # RR
            {'pos': [-0.19,  0.142, 0.0], 'force': [0.0, 0.0, 20.0]}  # RL
        ]

        self.start_time = time.time()
        
        # Publish at 50Hz (0.02s)
        self.timer = self.create_timer(0.02, self.publish_trajectory)
        self.get_logger().info("Publishing circular base trajectory...")

    def publish_trajectory(self):
        msg = MpcTrajectory()
        # msg.steps = [] # Assuming it's an array of steps
        
        t_current = time.time() - self.start_time
        omega = 2.0 * math.pi * self.freq

        for i in range(self.horizon_length):
            t = t_current + (i * self.dt)
            
            # Create your specific step message type here
            step = MpcStateStep() 
            
            # We will use a generic object simulation here - adapt to your ROS2 fields
            # 1. Base Kinematics
            step_pos_x = self.radius * math.cos(omega * t)
            step_pos_y = self.radius * math.sin(omega * t)
            
            step_vel_x = -self.radius * omega * math.sin(omega * t)
            step_vel_y =  self.radius * omega * math.cos(omega * t)
            
            step_acc_x = -self.radius * (omega**2) * math.cos(omega * t)
            step_acc_y = -self.radius * (omega**2) * math.sin(omega * t)

            # Assign Base Pose
            step.body_pose.position.x = step_pos_x
            step.body_pose.position.y = step_pos_y
            step.body_pose.position.z = self.z_height
            step.body_pose.orientation.w = 1.0  # Zero rotation
            
            # Assign Base Velocity
            step.body_velocity.linear.x = step_vel_x
            step.body_velocity.linear.y = step_vel_y
            step.body_velocity.linear.z = 0.0
            step.body_velocity.angular.x = 0.0
            step.body_velocity.angular.y = 0.0
            step.body_velocity.angular.z = 0.0
            
            # Assign Base Acceleration
            step.body_acceleration.linear.x = step_acc_x
            step.body_acceleration.linear.y = step_acc_y
            step.body_acceleration.linear.z = 0.0
            step.body_acceleration.angular.x = 0.0
            step.body_acceleration.angular.y = 0.0
            step.body_acceleration.angular.z = 0.0

            # 2. Feet 
            for j in range(4):
                foot = FootState()
                foot.position.x = self.nom_feet[j]['pos'][0]
                foot.position.y = self.nom_feet[j]['pos'][1]
                foot.position.z = self.nom_feet[j]['pos'][2]
                
                foot.velocity.x = 0.0
                foot.velocity.y = 0.0
                foot.velocity.z = 0.0
            
                foot.contact_force.x = self.nom_feet[j]['force'][0]
                foot.contact_force.y = self.nom_feet[j]['force'][1]
                foot.contact_force.z = self.nom_feet[j]['force'][2]
            
                foot.is_contact = True
                step.feet.append(foot)
            
            msg.steps.append(step)
        
        self.traj_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircularTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()