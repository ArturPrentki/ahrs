#!/usr/bin/env python3

import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from controller_manager_msgs.srv import SwitchController

# --- Configuration ---
CMD_VEL_TOPIC = '/cmd_vel'

# Define the exact names of your controllers as defined in your ros2_control YAML
GAIT_CONTROLLERS = {
    '1': 'gait_controller',
}

msg = """
WBC Robot Teleop Controller (Controller Manager Integration)
----------------------------------------------------------
Movement:
        w                  i (Z+)
   a    s    d       j (R-)      l (R+)
        x                  k (Z-)

w/x : Linear X      q/e : Yaw
a/d : Linear Y      j/l : Roll
i/k : Linear Z      u/o : Pitch

s   : Force Stop (Zero all velocities)

Swap Gait Controllers:
1   : gait_controller

Speed Control:
+ / - : Max Linear Speed +/- 10%
* / / : Max Angular Speed +/- 10%

CTRL-C to quit
"""

moveBindings = {
    'w': ( 1.0,  0.0,  0.0,  0.0,  0.0,  0.0), 'x': (-1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    'a': ( 0.0,  1.0,  0.0,  0.0,  0.0,  0.0), 'd': ( 0.0, -1.0,  0.0,  0.0,  0.0,  0.0),
    'i': ( 0.0,  0.0,  1.0,  0.0,  0.0,  0.0), 'k': ( 0.0,  0.0, -1.0,  0.0,  0.0,  0.0),
    'j': ( 0.0,  0.0,  0.0, -1.0,  0.0,  0.0), 'l': ( 0.0,  0.0,  0.0,  1.0,  0.0,  0.0),
    'u': ( 0.0,  0.0,  0.0,  0.0, -1.0,  0.0), 'o': ( 0.0,  0.0,  0.0,  0.0,  1.0,  0.0),
    'q': ( 0.0,  0.0,  0.0,  0.0,  0.0,  1.0), 'e': ( 0.0,  0.0,  0.0,  0.0,  0.0, -1.0),
}

speedBindings = {
    '+': (1.1, 1.0), '-': (0.9, 1.0),
    '*': (1.0, 1.1), '/': (1.0, 0.9),
}

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class WBCTeleop(Node):
    def __init__(self):
        super().__init__('wbc_teleop')
        self.pub_twist = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        
        # Service client to talk to the ros2_control controller manager
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        
        # Track the currently active gait so we know which one to deactivate
        self.current_gait = GAIT_CONTROLLERS['1']
        
    def switch_gait_controller(self, new_gait_name):
        if new_gait_name == self.current_gait:
            return # Already active
            
        if not self.switch_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Controller Manager service not available!")
            return

        req = SwitchController.Request()
        req.activate_controllers = [new_gait_name]
        req.deactivate_controllers = [self.current_gait]
        req.strictness = SwitchController.Request.STRICT

        future = self.switch_client.call_async(req)
        future.add_done_callback(lambda f: self.switch_callback(f, new_gait_name))

    def switch_callback(self, future, new_gait_name):
        try:
            response = future.result()
            if response.ok:
                print(f"\r\n[SUCCESS] Swapped to {new_gait_name}")
                self.current_gait = new_gait_name
            else:
                print(f"\r\n[FAILED] Could not swap to {new_gait_name}. Is it loaded?")
        except Exception as e:
            print(f"\r\nService call failed: {e}")

    def activate_wbc(self):
        # Wait for service is critical here, as this is called immediately on boot
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            print("\r\n[ERROR] Controller Manager service not available! Cannot activate WBC.")
            return

        req = SwitchController.Request()
        req.activate_controllers = ["optimal_controller", "gait_controller"]
        req.deactivate_controllers = ["stand_up"]
        req.strictness = SwitchController.Request.STRICT

        future = self.switch_client.call_async(req)
        # Added a callback to monitor success/failure
        future.add_done_callback(self.activate_wbc_callback)
        
    def activate_wbc_callback(self, future):
        try:
            response = future.result()
            if response.ok:
                print("\r\n[SUCCESS] Activated optimal_controller and gait_controller.")
            else:
                print("\r\n[FAILED] Could not activate WBC controllers. Make sure the launch file has fully finished loading them!")
        except Exception as e:
            print(f"\r\nService call failed: {e}")

    def deactivate_wbc(self):
        if not self.switch_client.wait_for_service(timeout_sec=2.0):
            return

        req = SwitchController.Request()
        req.deactivate_controllers = ["optimal_controller", "gait_controller"]
        req.activate_controllers = ["stand_up"]
        req.strictness = SwitchController.Request.STRICT

        future = self.switch_client.call_async(req)
        # Optional: you could add a callback here too if needed
                
def main(args=None):
    rclpy.init(args=args)
    node = WBCTeleop()
    settings = termios.tcgetattr(sys.stdin)

    linear_speed = 0.5
    angular_speed = 1.0
    x = y = z = th_x = th_y = th_z = 0.0
    
    print("Initializing teleop and connecting to Controller Manager...")
    node.activate_wbc()

    try:
        print(msg)
        while True:
            key = getKey(settings)

            if key in moveBindings.keys():
                x, y, z, th_x, th_y, th_z = moveBindings[key]
            
            elif key in speedBindings.keys():
                linear_speed *= speedBindings[key][0]
                angular_speed *= speedBindings[key][1]
                print(f"\rCurrent speeds - Linear: {linear_speed:.2f} Angular: {angular_speed:.2f}")

            elif key in GAIT_CONTROLLERS.keys():
                target_controller = GAIT_CONTROLLERS[key]
                print(f"\rRequesting switch from {node.current_gait} to {target_controller}...")
                node.switch_gait_controller(target_controller)

            elif key == 's':
                x = y = z = th_x = th_y = th_z = 0.0
                
            else:
                x = y = z = th_x = th_y = th_z = 0.0
                if (key == '\x03'): # CTRL-C
                    print("\r\nShutting down... reverting to stand_up controller.")
                    node.deactivate_wbc()
                    break

            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = x * linear_speed, y * linear_speed, z * linear_speed
            twist.angular.x, twist.angular.y, twist.angular.z = th_x * angular_speed, th_y * angular_speed, th_z * angular_speed
            node.pub_twist.publish(twist)

            # Process ROS callbacks (allows the service future to resolve)
            rclpy.spin_once(node, timeout_sec=0.01)

    except Exception as e:
        print(e)
    finally:
        twist = Twist()
        node.pub_twist.publish(twist) # Send zero velocity on shutdown
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()