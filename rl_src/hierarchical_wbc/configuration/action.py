import torch
from isaaclab.managers.action_manager import ActionTermCfg, ActionTerm

import warp as wp
from kinematics import BatchedQuadrupedIK
from isaaclab.utils import configclass
import torch
from isaaclab.managers.action_manager import ActionTermCfg, ActionTerm
from isaaclab.utils.math import yaw_quat, quat_mul, quat_conjugate, quat_apply
from kinematics import BatchedQuadrupedIK
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
class WbcSurrogateAction(ActionTerm):
    asset_name: str = "robot"
    
    def __init__(self, cfg: "WbcTaskSpaceActionCfg", env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 13, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 13, device=self.device)
        self.nominal_base_z = self.cfg.nominal_base_z
        
        # FIX: Assign tensors directly without register_buffer
        self.nominal_foot_pos = torch.tensor(self.cfg.nominal_foot_pos, device=self.device)
        self.HIP_OFFSETS_BASE_FRAME = torch.tensor([
            [ 0.1934,  0.0470, 0.0],  # FL
            [ 0.1934, -0.0470, 0.0],  # FR
            [-0.1934,  0.0470, 0.0],  # RL
            [-0.1934, -0.0470, 0.0]   # RR
        ], device=self.device)
        
        self.ik = BatchedQuadrupedIK(device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        # Calculate Target Base Height
        self.target_base_heights = self.nominal_base_z + (actions[:, 0] * 0.1)
        
        # Extract foot offsets
        foot_offsets = actions[:, 1:13].view(self.num_envs, 4, 3) * 0.2
        
        # Construct target foot positions IN THE GROUND HEADING FRAME
        self.target_foot_positions_ground_frame = self.nominal_foot_pos.clone() + foot_offsets

    def apply_actions(self):
        """Transform Ground Heading Frame -> CoM Heading Frame -> Base Frame -> Hip Frame"""
        # Translate Ground to CoM 
        vector_com_to_foot_heading = self.target_foot_positions_ground_frame.clone()
        vector_com_to_foot_heading[:, :, 2] -= self.target_base_heights.unsqueeze(-1)
        
        # Get Transformation from Heading Frame to Base Frame directly from Tensor
        # base_quat_w = self._env.scene["robot"].data.root_quat_w
        base_quat_w = wp.to_torch(self._env.scene["robot"].data.root_quat_w)
        yaw_q = yaw_quat(base_quat_w) 
        
        q_heading_to_base = quat_mul(quat_conjugate(base_quat_w), yaw_q)
        
        # Rotate foot targets into Base Frame
        flat_vector_heading = vector_com_to_foot_heading.view(-1, 3)
        flat_q_h2b = q_heading_to_base.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        flat_foot_pos_base = quat_apply(flat_q_h2b, flat_vector_heading)
        foot_pos_base_frame = flat_foot_pos_base.view(self.num_envs, 4, 3)
        
        # Transform to Hip Frame and solve IK (Broadcasting hip offsets automatically)
        foot_pos_hip_frame = foot_pos_base_frame - self.HIP_OFFSETS_BASE_FRAME
        target_joint_positions = self.ik.solve(foot_pos_hip_frame)
        
        # Apply joint targets
        self._env.scene["robot"].set_joint_position_target(target_joint_positions)

    @property
    def action_dim(self) -> int: return 13
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions

@configclass
class WbcTaskSpaceActionCfg(ActionTermCfg):
    class_type = WbcSurrogateAction
    asset_name: str = "robot"
    # You can pass config variables to the action term here
    nominal_base_z: float = 0.30 
    nominal_foot_pos = ([ 0.23,  0.10, 0.0],  # FL
                        [ 0.23, -0.10, 0.0],  # FR
                        [-0.23,  0.10, 0.0],  # RL
                        [-0.23, -0.10, 0.0])  # RR 
@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    task_space: WbcTaskSpaceActionCfg = WbcTaskSpaceActionCfg()