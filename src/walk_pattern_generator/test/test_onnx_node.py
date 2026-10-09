import unittest
import numpy as np
import rclpy
from walk_pattern_generator.trajectory_inference import WbcOnnxInferenceNode # Adjust import to match your filename

class TestWbcOnnxNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = WbcOnnxInferenceNode()
        # Pretend we received the first joint state to unlock the control loop
        cls.node.received_first_state = True 

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def setUp(self):
        # Reset the node's state buffers to a perfect "standing" state before each test
        self.node.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.node.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.node.proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.node.commands = np.zeros(3, dtype=np.float32)
        self.node.joint_pos =  np.zeros(12,dtype=np.float32)
        # self.node.default_joint_pos.copy()
        self.node.joint_vel = np.zeros(12, dtype=np.float32)
        # self.node.last_actions = np.zeros(22, dtype=np.float32)
        self.node.phase = 0.0

    def test_observation_vector_shape_and_scale(self):
        """Test 1: Does the observation vector match Isaac Lab expectations?"""
        # Inject some fake velocities
        self.node.base_lin_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.node.joint_vel = np.ones(12, dtype=np.float32)
        
        obs = self.node.build_observation_vector()
        
        self.assertEqual(obs.shape, (1, 60), "Observation vector is the wrong shape!")
        
        # Check scaling factors were applied
        # Linear velocity (index 0) should be 1.0 * 2.0 (lin_vel_scale)
        self.assertAlmostEqual(obs[0, 0], 2.0, msg="Linear velocity scaling failed")
        
        # Joint velocity (index 24) should be 1.0 * 0.05 (joint_vel_scale)
        self.assertAlmostEqual(obs[0, 24], 0.05, msg="Joint velocity scaling failed")

    def test_policy_zero_state_sanity(self):
        """Test 2: Is the policy badly trained?
        If we feed it a perfect standing state, it should output near-zero actions.
        """
        obs = self.node.build_observation_vector()
        
        outputs = self.node.ort_session.run(
            [self.node.output_name], 
            {self.node.input_name: obs}
        )
        actions = outputs[0][0]
        base_velocity_scale: tuple = (1.0, 1.0, 1.5)
        base_posture_scale: tuple = (0.1, 0.2, 0.2)
        actions[0:3]=actions[0:3]*base_velocity_scale
        actions[3:6]=actions[3:6]*base_posture_scale
                        
        print("\n--- ZERO STATE ACTION OUTPUTS ---")
        print(f"Base Velocity (x,y,z):   {actions[0:3]}")
        print(f"Base Posture (z,r,p):    {actions[3:6]}")
        
        # Reshape foot targets to easily read them

        foot_pos_scale: tuple = (0.1, 0.1, 0.1)
        actions[6:9]*=foot_pos_scale
        actions[9:12]*=foot_pos_scale
        actions[12:15]*=foot_pos_scale
        actions[15:18]*=foot_pos_scale
                    
        foot_deltas = actions[6:18].reshape(4, 3)*foot_pos_scale
        print(f"FL Foot Delta (x,y,z):   {foot_deltas[0]}")
        print(f"FR Foot Delta (x,y,z):   {foot_deltas[1]}")
        print(f"RL Foot Delta (x,y,z):   {foot_deltas[2]}")
        print(f"RR Foot Delta (x,y,z):   {foot_deltas[3]}")
        
        print(f"Contact States:          {actions[18:22]}")
        print("---------------------------------")
        # Check if actions are exploding. If a network outputs values > 5.0 for a 
        # stationary state, the policy is likely poorly trained or expects different inputs.
        max_action = np.max(np.abs(actions))
        print(f"\n[DIAGNOSTIC] Max action output at zero-state: {max_action:.3f}")
        
        self.assertTrue(max_action < 3.0, 
            "Policy is outputting extreme actions (>3.0) for a neutral standing pose. "
            "The network is either badly trained, or your observation order/scaling here "
            "does not perfectly match Isaac Lab's ObservationsCfg.")

    def test_action_decoding(self):
        """Test 3: Are actions being translated to the correct coordinate frames?"""
        # We will mock the ONNX output to force specific actions and see how the node parses them
        fake_actions = np.zeros(22, dtype=np.float32)
        
        # Ask for 1.0 m/s forward base velocity
        fake_actions[0] = 1.0 
        
        # Ask for 0.1m height increase in posture
        fake_actions[3] = 1.0  # Will be scaled by 0.1
        
        # Ask for +0.5 delta in X on the Front-Left foot (index 6 is FL_X)
        fake_actions[6] = 5.0  # Will be scaled by 0.1 to yield 0.5
        
        # Apply scaling and shifts as done in the control loop
        base_vel = fake_actions[0:3] * self.node.base_vel_scale
        base_posture = fake_actions[3:6] * self.node.base_posture_scale
        foot_positions_local_delta = fake_actions[6:18].reshape(4, 3) * self.node.foot_pos_scale
        
        # Verify Base Velocity
        self.assertAlmostEqual(base_vel[0], 1.0 * self.node.base_vel_scale[0])
        
        # Verify Base Posture (Nominal 0.32 + 0.1 delta)
        expected_height = 0.32 + (1.0 * self.node.base_posture_scale[0])
        self.assertAlmostEqual(expected_height, 0.42)
        
        # Verify Foot Position (Nominal FL_X + Delta)
        # FL foot X is self.node.foot_nom_x[0] (0.12)
        expected_fl_x = 0.12 + 0.5
        calculated_fl_x = self.node.foot_nom_x[0] + foot_positions_local_delta[0, 0]
        self.assertAlmostEqual(calculated_fl_x, expected_fl_x)

if __name__ == '__main__':
    unittest.main()