#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

msg = """
Quadruped Teleop Controller
---------------------------
Moving around:
    w
a   s   d

q/e : turn left/right
w/s : forward/backward
a/d : strafe left/right

spacebar : force stop
CTRL-C to quit
"""

moveBindings = {
    'w': (1.0, 0.0, 0.0),   # Forward
    's': (-1.0, 0.0, 0.0),  # Backward
    'a': (0.0, 1.0, 0.0),   # Strafe Left
    'd': (0.0, -1.0, 0.0),  # Strafe Right
    'q': (0.0, 0.0, 1.0),   # Turn Left
    'e': (0.0, 0.0, -1.0),  # Turn Right
}

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('quadruped_teleop')
        # Change '/cmd_vel' if your WalkController subscribes to a different topic
        self.publisher_ = self.create_publisher(Twist, '/vel_cmd', 10)
        self.speed = 0.50       # Base speed (m/s)
        self.turn_speed = 1.  # Base turn speed (rad/s)

    def publish_twist(self, x, y, th):
        twist = Twist()
        twist.linear.x = x * self.speed
        twist.linear.y = y * self.speed
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = th * self.turn_speed
        self.publisher_.publish(twist)

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main(args=None):
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=args)
    node = KeyboardTeleop()

    x = 0.0
    y = 0.0
    th = 0.0
    
    print(msg)

    try:
        while True:
            key = getKey(settings)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                th = moveBindings[key][2]
            elif key == ' ':
                x = 0.0
                y = 0.0
                th = 0.0
            else:
                x = 0.0
                y = 0.0
                th = 0.0
                if (key == '\x03'): # CTRL-C
                    break
                    
            node.publish_twist(x, y, th)

    except Exception as e:
        print(e)

    finally:
        node.publish_twist(0.0, 0.0, 0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()