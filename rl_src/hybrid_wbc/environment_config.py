from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import DomeLightCfg, SimulationCfg
from isaaclab.assets import AssetBaseCfg

from isaaclab.utils import configclass
from isaaclab_assets import UNITREE_GO2_CFG 

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnv
import sys
sys.path.append("/gardo_ws/")
import wbc_ext 

import torch
import numpy as np
from custom_rewards import *
from custom_observations import *
from custom_terminations import *
import os

@configclass
class Go2WbcSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
    
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_GO2_CFG.init_state.replace(
            joint_pos={
                "FL_hip_joint": 0.1,  "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
                "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
                "RL_hip_joint": 0.1,  "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
                "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
            }
        ),
    )
    # Zero out built-in PD gains for pure torque WBC control
    robot.actuators["base_legs"].stiffness = 0.0
    robot.actuators["base_legs"].damping = 0.0
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3,
        track_air_time=True,
    )


@configclass
class CommandsCfg:
    """Velocity commands locked to 0.0 for in-place trotting."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0, # Force standing/in-place
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), # Locked to 0
            lin_vel_y=(0.0, 0.0), # Locked to 0
            ang_vel_z=(0.0, 0.0), # Locked to 0
            heading=(0.0, 0.0),
        ),
    )

class WbcActionTerm(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        self.foot_pos_targets = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.base_posture_targets = torch.zeros(self.num_envs, 3, device=self.device)
        self.contact_targets = torch.zeros(self.num_envs, 4, device=self.device) 

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.trot_freq = 2.5 # Slightly slower trot is easier for initial WBC balance
        
        self.nominal_foot_pos = torch.tensor([
            [ 0.23,  0.10, 0.0],  # FL
            [ 0.23, -0.10, 0.0],  # FR
            [-0.01,  0.10, 0.0],  # RL
            [-0.01, -0.10, 0.0]   # RR 
        ], device=self.device)
        
        self.base_posture_scale = torch.tensor(self.cfg.base_posture_scale, device=self.device)
        self.foot_pos_scale = torch.tensor(self.cfg.foot_pos_scale, device=self.device)
        self.leg_phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.swing_height = 0.08

    @property
    def action_dim(self) -> int: return 15
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        # 1. Posture & Foot Deltas
        self.base_posture_targets = actions[:, 0:3] * self.base_posture_scale
        foot_deltas = actions[:, 3:15].view(self.num_envs, 4, 3) * self.foot_pos_scale

        # 2. Advance clock (using RL dt: 0.02s)
        # Note: self._env.step_dt is now 0.02 (dt * decimation)
        self.phase = (self.phase + self.trot_freq * self._env.step_dt) % 1.0
        leg_phases = (self.phase.unsqueeze(-1) + self.leg_phase_offsets) % 1.0 

        is_swing = leg_phases < 0.5
        swing_offset = torch.zeros(self.num_envs, 4, 3, device=self.device)
        swing_offset[..., 2] = torch.where(
            is_swing, 
            self.swing_height * torch.sin(2.0 * torch.pi * leg_phases), 
            0.0
        )

        self.foot_pos_targets = self.nominal_foot_pos + swing_offset + foot_deltas
        self.contact_targets = (~is_swing).float()

    def apply_actions(self):
        # DO NOTHING HERE. 
        # We will apply torques inside the custom physics loop.
        pass
    def reset(self, env_ids: torch.Tensor):
        self.foot_pos_targets[env_ids] = self.nominal_foot_pos
        self.base_posture_targets[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.phase[env_ids] = 0.0

class WbcActionTerm_old(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        self.foot_pos_targets = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.base_posture_targets = torch.zeros(self.num_envs, 3, device=self.device)
        self.contact_targets = torch.zeros(self.num_envs, 4, device=self.device) 

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.trot_freq = 2.5 # Slightly slower trot is easier for initial WBC balance
        
        self.nominal_foot_pos = torch.tensor([
            [ 0.23,  0.10, 0.0],  # FL
            [ 0.23, -0.10, 0.0],  # FR
            [-0.01,  0.10, 0.0],  # RL
            [-0.01, -0.10, 0.0]   # RR 
        ], device=self.device)
        
        self.base_posture_scale = torch.tensor(self.cfg.base_posture_scale, device=self.device)
        self.foot_pos_scale = torch.tensor(self.cfg.foot_pos_scale, device=self.device)
        self.leg_phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.swing_height = 0.08

        urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "go2_description.urdf") 
        with open(urdf_path, "r") as f:
            urdf_string = f.read()
            
        self.wbc = wbc_ext.BatchedWBC(self.num_envs, urdf_string)

    @property
    def action_dim(self) -> int: return 15
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        
        self.base_posture_targets = actions[:, 0:3] * self.base_posture_scale
        foot_deltas = actions[:, 3:15].view(self.num_envs, 4, 3) * self.foot_pos_scale

        # Advance clock 
        self.phase = (self.phase + self.trot_freq * self._env.step_dt) % 1.0
        leg_phases = (self.phase.unsqueeze(-1) + self.leg_phase_offsets) % 1.0 

        is_swing = leg_phases < 0.5
        swing_offset = torch.zeros(self.num_envs, 4, 3, device=self.device)
        # Sine wave trajectory mapping 0.0->0.5 phase to 0->swing_height->0
        swing_offset[..., 2] = torch.where(
            is_swing, 
            self.swing_height * torch.sin(2.0 * torch.pi * leg_phases), 
            0.0
        )

        self.foot_pos_targets = self.nominal_foot_pos + swing_offset + foot_deltas
        self.contact_targets = (~is_swing).float()
        
        self._processed_actions[:] = actions

    def apply_actions(self):
        base_pos = self._asset.data.root_pos_w       
        base_quat_wxyz = self._asset.data.root_quat_w 
        base_lin_vel = self._asset.data.root_lin_vel_w 
        base_ang_vel = self._asset.data.root_ang_vel_w 
        
        joint_pos = self._asset.data.joint_pos      
        joint_vel = self._asset.data.joint_vel       
        
        contact_sensor = self._env.scene.sensors["contact_forces"]
        foot_indices = contact_sensor.find_bodies(".*_foot")[0] 
        
        # NOTE: Verify foot_indices shape aligns exactly with [FL, FR, RL, RR] in your WBC
        foot_forces = contact_sensor.data.net_forces_w[:, foot_indices, :] 
        # Increased contact threshold to 5.0N to filter out simulation noise
        actual_contacts = (torch.norm(foot_forces, dim=-1) > 5.0).float().unsqueeze(-1) 

        desired_vel = self._env.command_manager.get_command("base_velocity")[:, 0:3]

        j_pos_reshaped = joint_pos.view(self.num_envs, 4, 3)
        j_vel_reshaped = joint_vel.view(self.num_envs, 4, 3)
        leg_states = torch.cat([j_pos_reshaped, j_vel_reshaped, actual_contacts], dim=-1)
        leg_states_flat = leg_states.view(self.num_envs, 28) 

        observations = torch.cat([
            base_pos, base_quat_wxyz, base_lin_vel, base_ang_vel, leg_states_flat
        ], dim=1)

        actions = torch.cat([
            self.base_posture_targets,                     
            self.foot_pos_targets.view(self.num_envs, 12), 
            self.contact_targets                           
        ], dim=1)

        torques = self.wbc.solve(
            observations=observations,
            actions=actions,
            commands=desired_vel,
            dt=self._env.step_dt
        )
        
        self._asset.set_joint_effort_target(torques.to(device=self.device, dtype=torch.float32).contiguous())

    def reset(self, env_ids: torch.Tensor):
        self.foot_pos_targets[env_ids] = self.nominal_foot_pos
        self.base_posture_targets[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.phase[env_ids] = 0.0


@configclass
class WbcActionCfg(ActionTermCfg):
    class_type: type = WbcActionTerm
    asset_name: str = "robot"
    base_posture_scale: tuple = (0.05, 0.1, 0.1)
    foot_pos_scale: tuple = (0.05, 0.05, 0.05) # Reduced scale to prevent wild exploration


@configclass
class ActionsCfg:
    wbc_command = WbcActionCfg()

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        foot_positions = ObsTerm(func=foot_positions_in_base_frame, params={"asset_name": "robot"})
        phase = ObsTerm(func=phase_clock_observation, params={"action_term_name": "wbc_command"})
        contact_states = ObsTerm(func=physical_contact_states, params={"sensor_name": "contact_forces"})
        last_action = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 8
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    # Task Tracking
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.5, params={"command_name": "base_velocity", "std": 0.25})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5})

    # Posture & Stability
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.30})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)

    # WBC Smoothness 
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    
    # HEAVILY penalize RL action magnitude so it relies on the CPG first
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.5) 
    
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)

    # Gait Custom Rewards
    knee_scraping = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]), "threshold": 0.1})
    gait_following = RewTerm(func=trot_gait_reward, weight=1.5, params={"action_term_name": "wbc_command"})
    foot_spacing_penalty = RewTerm(func=wbc_foot_clearance_penalty, weight=-1.0, params={"action_term_name": "wbc_command", "min_lateral_dist": 0.12, "min_longitudinal_dist": 0.15})
    slipping_foot_penalty = RewTerm(func=foot_slip_penalty, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]), "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"])})

@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        },
    )
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale, mode="reset", params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)})
    push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0), params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}})

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.5})


@configclass
class Go2WbcEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2WbcSceneCfg = Go2WbcSceneCfg(num_envs=256, env_spacing=2.5)
    
    sim: SimulationCfg = SimulationCfg(
        dt=0.001,                 # Physics step size (1000 Hz)
        render_interval=20)

    decimation: int = 20
    
    episode_length_s: float = 20.0
    
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


class Go2WbcEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: Go2WbcEnvCfg):
        super().__init__(cfg)
        
        # Initialize WBC here
        urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "go2_description.urdf") 
        with open(urdf_path, "r") as f:
            urdf_string = f.read()
        self.wbc = wbc_ext.BatchedWBC(self.num_envs, urdf_string)
        
        # Get references to the asset and action term for fast access in the loop
        self.robot = self.scene.articulations["robot"]
        self.wbc_action = self.action_manager.action_terms["wbc_command"]
        self.contact_sensor = self.scene.sensors["contact_forces"]
        self.foot_indices = self.contact_sensor.find_bodies(".*_foot")[0]

    def _advance_simulation(self):
        """Runs the 1000Hz physics loop, executing the WBC at every step."""
        
        # Loop exactly 'decimation' times (20 times)
        for _ in range(self.cfg.decimation):
            
            # --- 1. Gather 1000Hz Simulator State ---
            base_pos = self.robot.data.root_pos_w       
            base_quat = self.robot.data.root_quat_w 
            base_lin_vel = self.robot.data.root_lin_vel_w #available outside the gpu
            base_ang_vel = self.robot.data.root_ang_vel_w #available outside the gpu
            
            joint_pos = self.robot.data.joint_pos      
            joint_vel = self.robot.data.joint_vel       
            
            foot_forces = self.contact_sensor.data.net_forces_w[:, self.foot_indices, :] 
            actual_contacts = (torch.norm(foot_forces, dim=-1) > 5.0).float().unsqueeze(-1) 

            # --- 2. Format Tensors ---
            leg_states = torch.cat([
                joint_pos.view(self.num_envs, 4, 3), 
                joint_vel.view(self.num_envs, 4, 3), 
                actual_contacts
            ], dim=-1).view(self.num_envs, 28) 

            observations = torch.cat([
                base_pos, base_quat, base_lin_vel, base_ang_vel, leg_states
            ], dim=1)

            # --- 3. Pull 50Hz Targets from Action Term ---
            actions = torch.cat([
                self.wbc_action.base_posture_targets,                     
                self.wbc_action.foot_pos_targets.view(self.num_envs, 12), 
                self.wbc_action.contact_targets                           
            ], dim=1)

            commands = self.command_manager.get_command("base_velocity")[:, 0:3]

            # --- 4. Call WBC at 1000Hz ---
            # Use physics dt (0.001) for the WBC math, not the RL dt (0.02)
            torques = self.wbc.solve(
                observations=observations,
                actions=actions,
                commands=commands,
                dt=self.physics_dt 
            )
            
            # --- 5. Apply and Step Physics ---
            self.robot.set_joint_effort_target(torques.to(device=self.device, dtype=torch.float32).contiguous())
            
            # Step the Isaac Sim physics engine by 1 dt
            self.sim.step()

            