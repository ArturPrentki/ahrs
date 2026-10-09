from isaaclab.managers import RewardTermCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
import torch

def swing_phase_contact_penalty(
    env, 
    contact_sensor_name: str, 
    command_name: str, 
    foot_body_names: list[str] | str = ".*foot"
) -> torch.Tensor:
    """Penalizes foot contacts when phase clock indicates the foot should be swinging."""
    phase = env.command_manager.get_term(command_name)._clock  # [num_envs, 4]
    contact_sensor = env.scene.sensors[contact_sensor_name]

    if not hasattr(contact_sensor, "_cached_foot_tensor_ids"):
        foot_ids, _ = contact_sensor.find_bodies(foot_body_names)
        # Convert to tensor and push to device ONCE
        contact_sensor._cached_foot_tensor_ids = torch.tensor(
            foot_ids, device=env.device, dtype=torch.long
        )
    foot_tensor_ids = contact_sensor._cached_foot_tensor_ids
    contact_forces = contact_sensor.data.net_forces_w[:, foot_tensor_ids, 2]
    
    is_swinging = (phase > 0.5).float()
    contact_during_swing = (contact_forces > 1.0).float()
    
    return torch.sum(is_swinging * contact_during_swing, dim=1)

def stance_phase_contact_reward(
    env, 
    contact_sensor_name: str, 
    command_name: str,
    foot_body_names: list[str] | str = ".*foot"
) -> torch.Tensor:
    """Rewards maintaining contact with the ground when the phase clock indicates stance."""
    phase = env.command_manager.get_term(command_name)._clock
    contact_sensor = env.scene.sensors[contact_sensor_name]
    if not hasattr(contact_sensor, "_cached_foot_tensor_ids"):
        foot_ids, _ = contact_sensor.find_bodies(foot_body_names)
        contact_sensor._cached_foot_tensor_ids = torch.tensor(
            foot_ids, device=env.device, dtype=torch.long
        )
    foot_tensor_ids = contact_sensor._cached_foot_tensor_ids
    contact_forces = contact_sensor.data.net_forces_w[:, foot_tensor_ids, 2]
    is_stance = (phase <= 0.5).float()
    
    # Reward: Having contact force > 1.0 during stance
    # contact_during_stance = torch.where(contact_forces > 1.0, 1.0, 0.0)
    contact_during_stance = torch.clamp(contact_forces / 50.0, 0.0, 1.0)
    # Sum the rewards across all 4 legs
    return torch.sum(is_stance * contact_during_stance, dim=1)

def swing_height_tracking(
    env, 
    command_name: str, 
    target_height: float = 0.08,
    foot_body_names: list[str] | str = ".*foot"
) -> torch.Tensor:
    """Encourages feet to lift off the ground forming a trajectory during swing."""
    robot = env.scene["robot"]

    if not hasattr(robot, "_cached_foot_tensor_ids"):
        foot_ids, _ = robot.find_bodies(foot_body_names)
        robot._cached_foot_tensor_ids = torch.tensor(
            foot_ids, device=env.device, dtype=torch.long
        )
        
    foot_tensor_ids = robot._cached_foot_tensor_ids
    foot_positions = robot.data.body_pos_w[:, foot_tensor_ids, :]
    
    foot_z = foot_positions[..., 2]
    
    phase = env.command_manager.get_term(command_name)._clock
    swing_progress = (phase - 0.5) * 2.0 
    swing_progress = torch.clamp(swing_progress, 0.0, 1.0)
    
    foot_radius = 0.02
    desired_z = foot_radius + target_height * torch.sin(swing_progress * torch.pi)
    
    is_swinging = (phase > 0.5).float()
    
    # Penalize deviation from the expected swing height arch
    error_sq = torch.square(foot_z - desired_z)
    # With -10, a 5cm error only reduced the reward by 2%. 
    # With -100, a 5cm error reduces it by 22%, forcing the agent to track it closely.
    return torch.sum(is_swinging * torch.exp(-100.0 * error_sq), dim=1)

@configclass
class RewardsCfg:
    # 1. Base Velocity Tracking (The main goal)
    track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.1})
    track_ang_vel_z_exp = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.1})

    # 2. Gait Shaping Rewards (Forces the RL to output walking trajectories)
    gait_swing_penalty = RewardTermCfg(
        func=swing_phase_contact_penalty, 
        weight=-0.25, 
        params={
            "contact_sensor_name": "contact_forces",
            "command_name": "gait_clock", 
            "foot_body_names": [".*FL_foot", ".*FR_foot", ".*RL_foot", ".*RR_foot"],
        }
    )
    
    gait_stance_reward = RewardTermCfg(
        func=stance_phase_contact_reward, 
        weight=0.25, 
        params={
            "contact_sensor_name": "contact_forces",
            "command_name": "gait_clock",
            "foot_body_names": [".*FL_foot", ".*FR_foot", ".*RL_foot", ".*RR_foot"],
        }
    )
    
    gait_swing_height = RewardTermCfg(
        func=swing_height_tracking, 
        weight=1.5, 
        params={
            "command_name": "gait_clock", 
            "target_height": 0.08,
            "foot_body_names": [".*FL_foot", ".*FR_foot", ".*RL_foot", ".*RR_foot"],
        }
    )

    # 3. Posture and Smoothness
    base_height_l2 = RewardTermCfg(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.30})
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)

    action_l2 = RewardTermCfg(func=mdp.action_l2, weight=-0.5)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01) 

