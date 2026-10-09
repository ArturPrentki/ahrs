import rclpy
from rclpy.node import Node
import math
import time

from wbc_control_msgs.msg import MpcTrajectory, MpcStateStep, FootState 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class MpcRaibertTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('mpc_raibert_trajectory_generator')
        
        self.traj_pub = self.create_publisher(MpcTrajectory, '/mpc_trajectory', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_command = self.create_subscription(Twist, '/vel_cmd', self.command_callback, 10)
        
        self.horizon_length = 100
        self.dt = 0.001
        
        self.actual_x = 0.0
        self.actual_y = 0.0
        self.actual_yaw = 0.0

        self.v_actual_x = 0.0
        self.v_actual_y = 0.0
        
        self.v_cmd_x = 0.0
        self.v_cmd_y = 0.0
        self.w_cmd_yaw = 0.0  # NEW: Yaw rate command
        
        # Global base tracking now includes yaw
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_yaw = 0.0 
        
        self.start_time = None
        self.last_time = None
        
        self.z_height = 0.3
        self.foot_nom_x = [ 0.19,  0.19, -0.19, -0.19]  # FR, FL, RR, RL
        self.foot_nom_y = [-0.142, 0.142, -0.142, 0.142]
        
        self.gait_period = 0.4
        self.duty_factor = 0.5 
        self.swing_height = 0.05
        self.phase_offsets = [0.0, 0.5, 0.5, 0.0] 
        self.k_vel_feedback = 0.03 
        
        self.timer = self.create_timer(0.02, self.publish_trajectory)
        self.get_logger().info("Raibert MPC Trajectory Generator (with Yaw) Started.")

    def bezier_curve(self, p0, p1, p2, p3, t):
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    def odom_callback(self, msg):
        self.v_actual_x = msg.twist.twist.linear.x
        self.v_actual_y = msg.twist.twist.linear.y
        self.actual_x = msg.pose.pose.position.x
        self.actual_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.actual_yaw = math.atan2(siny_cosp, cosy_cosp)
    def command_callback(self, msg):
        self.v_cmd_x = msg.linear.x
        self.v_cmd_y = msg.linear.y
        self.w_cmd_yaw = msg.angular.z

    def integrate_pose(self, x, y, yaw, vx, vy, omega, dt):
        """Analytically integrates a local velocity command into a global frame."""
        if abs(omega) < 1e-3:
            # Linear integration if not turning
            nx = x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            ny = y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            nyaw = yaw
        else:
            # Exact arc integration if turning
            nyaw = yaw + omega * dt
            nx = x + (vx * (math.sin(nyaw) - math.sin(yaw)) + vy * (math.cos(nyaw) - math.cos(yaw))) / omega
            ny = y + (vx * (math.cos(yaw) - math.cos(nyaw)) + vy * (math.sin(nyaw) - math.sin(yaw))) / omega
        return nx, ny, nyaw

    def publish_trajectory(self):
        now = self.get_clock().now()
        
        if self.start_time is None:
            self.start_time = now
            self.last_time = now
            return
            
        dt_timer = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        
        # 1. Integrate the global base origin for this tick
        # self.base_x, self.base_y, self.base_yaw = self.integrate_pose(
        #     self.base_x, self.base_y, self.base_yaw, 
        #     self.v_cmd_x, self.v_cmd_y, self.w_cmd_yaw, dt_timer
        # )
        self.base_x = self.actual_x
        self.base_y = self.actual_y
        self.base_yaw = self.actual_yaw

        t_current_abs = (now - self.start_time).nanoseconds / 1e9
        
        msg = MpcTrajectory()
        T_stance = self.gait_period * self.duty_factor
        T_swing = self.gait_period * (1.0 - self.duty_factor)

        # 2. Generate the MPC Horizon
        for k in range(self.horizon_length):
            t_k = t_current_abs + (k * self.dt)
            dt_k = k * self.dt
            step = MpcStateStep()
            
            # --- Base Kinematics Prediction ---
            base_x_k, base_y_k, base_yaw_k = self.integrate_pose(
                self.base_x, self.base_y, self.base_yaw,
                self.v_cmd_x, self.v_cmd_y, self.w_cmd_yaw, dt_k
            )
            
            step.body_pose.position.x = base_x_k
            step.body_pose.position.y = base_y_k
            step.body_pose.position.z = self.z_height
            
            # Convert yaw to quaternion
            step.body_pose.orientation.x = 0.0
            step.body_pose.orientation.y = 0.0
            step.body_pose.orientation.z = math.sin(base_yaw_k / 2.0)
            step.body_pose.orientation.w = math.cos(base_yaw_k / 2.0)
            
            # Local velocities
            step.body_velocity.linear.x = self.v_cmd_x
            step.body_velocity.linear.y = self.v_cmd_y
            step.body_velocity.linear.z = 0.0
            step.body_velocity.angular.x = 0.0
            step.body_velocity.angular.y = 0.0
            step.body_velocity.angular.z = self.w_cmd_yaw
            
            # --- Feet Kinematics Prediction ---
            for i in range(4):
                foot = FootState()
                
                phi_total = (t_k / self.gait_period) + self.phase_offsets[i]
                cycle_num = math.floor(phi_total) 
                phase = phi_total - cycle_num     
                
                def get_stance_target(c_num):
                    # Find exact time of mid-stance for this step
                    t_mid = (c_num - self.phase_offsets[i]) * self.gait_period + (0.5 * T_stance)
                    dt_mid = t_mid - t_current_abs
                    
                    # Predict base pose at mid-stance
                    b_x_mid, b_y_mid, b_yaw_mid = self.integrate_pose(
                        self.base_x, self.base_y, self.base_yaw, 
                        self.v_cmd_x, self.v_cmd_y, self.w_cmd_yaw, dt_mid
                    )
                    
                    # # Compute local foot velocity due to robot rotation (v = w x r)
                    # v_foot_local_x = self.v_cmd_x - self.w_cmd_yaw * self.foot_nom_y[i]
                    # v_foot_local_y = self.v_cmd_y + self.w_cmd_yaw * self.foot_nom_x[i]
                    
                    # # Raibert target in the LOCAL frame
                    # r_x = self.foot_nom_x[i] + 0.5 * T_stance * v_foot_local_x + self.k_vel_feedback * (self.v_actual_x - self.v_cmd_x)
                    # r_y = self.foot_nom_y[i] + 0.5 * T_stance * v_foot_local_y + self.k_vel_feedback * (self.v_actual_y - self.v_cmd_y)
                    v_foot_actual_local_x = self.v_actual_x - self.w_cmd_yaw * self.foot_nom_y[i]
                    
                    v_foot_actual_local_y = self.v_actual_y + self.w_cmd_yaw * self.foot_nom_x[i]
                    r_x = self.foot_nom_x[i] + (0.5 * T_stance * v_foot_actual_local_x) + self.k_vel_feedback * (self.v_actual_x - self.v_cmd_x)
                    r_y = self.foot_nom_y[i] + (0.5 * T_stance * v_foot_actual_local_y) + self.k_vel_feedback * (self.v_actual_y - self.v_cmd_y)
                    
                    # Rotate the local target into the GLOBAL frame and add to base position
                    f_x = b_x_mid + (r_x * math.cos(b_yaw_mid) - r_y * math.sin(b_yaw_mid))
                    f_y = b_y_mid + (r_x * math.sin(b_yaw_mid) + r_y * math.cos(b_yaw_mid))
                    return (f_x, f_y)

                if phase < self.duty_factor:
                    # == STANCE PHASE ==
                    p_stance = get_stance_target(cycle_num)
                    
                    foot.position.x = p_stance[0]
                    foot.position.y = p_stance[1]
                    foot.position.z = 0.0
                    
                    foot.velocity.x = 0.0
                    foot.velocity.y = 0.0
                    foot.velocity.z = 0.0
                    
                    foot.contact_force.x = 0.0
                    foot.contact_force.y = 0.0
                    foot.contact_force.z = 60.0 # Applied dynamic 12kg feedforward logic
                    foot.is_contact = True
                    
                else:
                    # == SWING PHASE ==
                    s = (phase - self.duty_factor) / (1.0 - self.duty_factor)
                    
                    p_start = get_stance_target(cycle_num)
                    p_end = get_stance_target(cycle_num + 1)
                    
                    foot.position.x = float(self.bezier_curve(p_start[0], p_start[0], p_end[0], p_end[0], s))
                    foot.position.y = float(self.bezier_curve(p_start[1], p_start[1], p_end[1], p_end[1], s))
                    foot.position.z = float(self.bezier_curve(0.0, self.swing_height, self.swing_height, 0.0, s))
                    
                    dx_ds = 6.0 * s * (1.0 - s) * (p_end[0] - p_start[0])
                    dy_ds = 6.0 * s * (1.0 - s) * (p_end[1] - p_start[1])
                    dz_ds = 3.0 * self.swing_height * (1.0 - 2.0 * s)
                    
                    foot.velocity.x = dx_ds / T_swing
                    foot.velocity.y = dy_ds / T_swing
                    foot.velocity.z = dz_ds / T_swing
                    
                    foot.contact_force.x = 0.0
                    foot.contact_force.y = 0.0
                    foot.contact_force.z = 0.0
                    foot.is_contact = False
                    
                step.feet.append(foot)
            
            msg.steps.append(step)
            
        self.traj_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MpcRaibertTrajectoryGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
