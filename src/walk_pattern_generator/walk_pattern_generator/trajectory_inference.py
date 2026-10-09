from time import sleep

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

# Custom WBC messages
from wbc_control_msgs.msg import WbcVelocityCommand, FootPositionTarget

# Standard ROS 2 state messages
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class WbcOnnxInferenceNode(Node):
    def __init__(self):
        super().__init__('wbc_onnx_inference')
        self.REMOVE_ME_COUNTER = 10        
        
        # 1. Load ONNX Session
        model_path = "/gardo_ws/policies/policy.onnx"
        self.get_logger().info(f"Loading ONNX policy from {model_path}...")
        
        self.ort_session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # 2. Action Scaling Parameters
        self.base_vel_scale = np.array([1.0, 1.0, 1.5], dtype=np.float32)
        self.base_posture_scale = np.array([0.1, 0.2, 0.2], dtype=np.float32)
        self.foot_pos_scale = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        # 3. Observation Scale Factors
        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25
        self.joint_vel_scale = 0.05

        # Phase Clock Parameters
        self.trot_freq = 3.0  
        self.dt = 0.02         
        self.phase = 0.0       
        
        # Order: Hips(FL,FR,RL,RR), Thighs(FL,FR,RL,RR), Calves(FL,FR,RL,RR)
        self.default_joint_pos = np.array([
             0.1, -0.1,  0.1, -0.1,  
             0.8,  0.8,  1.0,  1.0,  
            -1.5, -1.5, -1.5, -1.5   
        ], dtype=np.float32)

        # EXACT match to nominal_foot_pos in training WbcActionTerm
        self.nominal_foot_pos = np.array([
            [ 0.23,  0.10, 0.0],  # FL
            [ 0.23, -0.10, 0.0],  # FR
            [-0.01,  0.10, 0.0],  # RL
            [-0.01, -0.10, 0.0]   # RR
        ], dtype=np.float32)
        
        # 4. State Buffers
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32) 
        self.joint_pos = self.default_joint_pos.copy()
        self.joint_vel = np.zeros(12, dtype=np.float32)
        
        # NEW BUFFERS for updated observations
        self.current_foot_pos = np.zeros(12, dtype=np.float32)
        self.current_contacts = np.ones(4, dtype=np.float32) # Default to 1 (touching ground)
        
        self.last_actions = np.zeros(22, dtype=np.float32)
        
        self.latest_rot = R.identity()

        # 5. ROS 2 Interfaces
        self.pub = self.create_publisher(WbcVelocityCommand, '/wbc_velocity_command', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.sub_joints = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # TODO: Add a subscriber here for your WBC's contact estimation / low_state foot forces
        # self.sub_contacts = self.create_subscription(..., '/foot_contacts', self.contact_callback, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("ONNX Inference Node initialized and running at 50Hz.")
        self.received_first_state = False

    def compute_go2_fk(self, q):
        """
        Computes analytical Forward Kinematics for Unitree Go2 legs.
        q expects shape [12]: 4 hips, 4 thighs, 4 calves
        Returns [12] flattened array of (X, Y, Z) for FL, FR, RL, RR in base frame.
        """
        feet = np.zeros(12, dtype=np.float32)
        
        # Go2 Kinematic parameters (approximate meters)
        hip_offsets = [
            np.array([ 0.1881,  0.04675, 0.0]), # FL
            np.array([ 0.1881, -0.04675, 0.0]), # FR
            np.array([-0.1881,  0.04675, 0.0]), # RL
            np.array([-0.1881, -0.04675, 0.0])  # RR
        ]
        hip_y = [0.0955, -0.0955, 0.0955, -0.0955] # L_hip
        L_thigh = 0.213
        L_calf = 0.213

        for i in range(4):
            qh = q[i]         # Hip
            qt = q[4 + i]     # Thigh
            qc = q[8 + i]     # Calf

            # Rotation from Hip joint (X axis)
            Rx = np.array([[1, 0, 0], 
                           [0, np.cos(qh), -np.sin(qh)], 
                           [0, np.sin(qh), np.cos(qh)]])
            
            # Rotation from Thigh joint (Y axis)
            R_thigh = np.array([[np.cos(qt), 0, np.sin(qt)], 
                                [0, 1, 0], 
                                [-np.sin(qt), 0, np.cos(qt)]])
            
            # Rotation from Calf joint (Y axis, cumulative)
            qt_c = qt + qc
            R_calf = np.array([[np.cos(qt_c), 0, np.sin(qt_c)], 
                               [0, 1, 0], 
                               [-np.sin(qt_c), 0, np.cos(qt_c)]])

            # Local vectors
            p_hip = np.array([0, hip_y[i], 0])
            p_thigh = np.array([0, 0, -L_thigh])
            p_calf = np.array([0, 0, -L_calf])

            # Forward kinematics chain
            pos = hip_offsets[i] + Rx @ (p_hip + R_thigh @ p_thigh + R_calf @ p_calf)
            feet[i*3 : i*3+3] = pos

        return feet

    def imu_callback(self, msg: Imu):
        self.base_ang_vel[0] = msg.angular_velocity.x
        self.base_ang_vel[1] = msg.angular_velocity.y
        self.base_ang_vel[2] = msg.angular_velocity.z

        q = msg.orientation
        self.latest_rot = R.from_quat([q.x, q.y, q.z, q.w])
        self.proj_gravity = self.latest_rot.inv().apply([0.0, 0.0, -1.0]).astype(np.float32)

    def odom_callback(self, msg: Odometry):
        self.base_lin_vel[0] = msg.twist.twist.linear.x
        self.base_lin_vel[1] = msg.twist.twist.linear.y
        self.base_lin_vel[2] = msg.twist.twist.linear.z

    def joint_callback(self, msg: JointState):
        target_names = [
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
        ]
        
        if len(msg.position) < 12:
            return
            
        try:
            for i, target_name in enumerate(target_names):
                msg_idx = msg.name.index(target_name)
                self.joint_pos[i] = msg.position[msg_idx]
                self.joint_vel[i] = msg.velocity[msg_idx]
            
            # Update Cartesian foot positions using FK
            self.current_foot_pos = self.compute_go2_fk(self.joint_pos)
            
            thigh_error = abs(self.joint_pos[4] - self.default_joint_pos[4])
            if thigh_error < 0.5: 
                self.received_first_state = True
                
        except ValueError:
            self.get_logger().error(f"Missing joint in message! Expected: {target_names}")
            
    def cmd_callback(self, msg: Twist):
        self.commands[0] = msg.linear.x
        self.commands[1] = msg.linear.y
        self.commands[2] = msg.angular.z

    def build_observation_vector(self) -> np.ndarray:
        joint_pos_delta = self.joint_pos - self.default_joint_pos

        phase_rad = self.phase * 2.0 * np.pi
        phase_obs = np.array([np.sin(phase_rad), np.cos(phase_rad)], dtype=np.float32)

        # Concatenate strictly matching the ObservationsCfg order in training
        obs = np.concatenate([
            self.base_lin_vel * self.lin_vel_scale,  
            self.base_ang_vel * self.ang_vel_scale,  
            self.proj_gravity,                       
            self.commands,                           
            joint_pos_delta,                         
            self.joint_vel * self.joint_vel_scale,   
            self.last_actions,                       
            phase_obs,
            self.current_foot_pos,      # NEW: 12D Cartesian Foot Pos
            self.current_contacts       # NEW: 4D Contact States
        ], axis=0).astype(np.float32)

        if self.REMOVE_ME_COUNTER > 0:
            print(f"Total obs size: {len(obs)} (Should be 76)")
            self.REMOVE_ME_COUNTER -= 1

        return np.expand_dims(obs, axis=0)

    def control_loop(self):
        if not self.received_first_state:
            return
        
        self.phase = (self.phase + self.trot_freq * self.dt) % 1.0

        obs_batch = self.build_observation_vector()
        
        outputs = self.ort_session.run(
            [self.output_name], 
            {self.input_name: obs_batch}
        )
        raw_actions = outputs[0][0]  

        # --- ACTION SMOOTHING (Low-Pass Filter) ---
        alpha = 0.2
        actions = alpha * raw_actions + (1.0 - alpha) * self.last_actions
        self.last_actions = actions.copy()

        # 1. Base Targets
        base_vel = actions[0:3] * self.base_vel_scale
        base_posture = actions[3:6] * self.base_posture_scale
        
        # 2. Foot Positions (FIXED: They are deltas!)
        foot_deltas = actions[6:18].reshape(4, 3) * self.foot_pos_scale
        foot_positions_base_frame = self.nominal_foot_pos + foot_deltas
        
        # 3. Contact States
        contact_states = actions[18:22] > 0.0

        nominal_height = 0.32
        base_z_target = float(nominal_height + base_posture[0])
        base_z_target = np.clip(base_z_target, 0.25, 0.40)

        # 4. Message Assembly
        msg = WbcVelocityCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_heading"

        msg.base_velocity.linear.x = float(base_vel[0])
        msg.base_velocity.linear.y = float(base_vel[1])
        msg.base_velocity.linear.z = 0.0
        msg.base_velocity.angular.z = float(base_vel[2])

        msg.base_height_target = base_z_target
        msg.base_roll_target = float(base_posture[1])
        msg.base_pitch_target = float(base_posture[2])
        
        feet = []
        for i in range(4):
            foot = FootPositionTarget()
            
            foot.position.x = float(foot_positions_base_frame[i, 0])
            foot.position.y = float(foot_positions_base_frame[i, 1])
            foot.position.z = float(foot_positions_base_frame[i, 2])
            
            foot.contact_state = bool(contact_states[i])
            feet.append(foot)
            
        msg.foot_targets = feet
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WbcOnnxInferenceNode()
    sleep(3)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()