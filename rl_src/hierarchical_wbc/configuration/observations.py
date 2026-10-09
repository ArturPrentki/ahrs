import torch
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

def phase_clock_sin(env, command_name: str) -> torch.Tensor:
    """Continuous sine representation of the phase clock."""
    phase = env.command_manager.get_command(command_name)
    return torch.sin(2.0 * torch.pi * phase)

def phase_clock_cos(env, command_name: str) -> torch.Tensor:
    """Continuous cosine representation of the phase clock."""
    phase = env.command_manager.get_command(command_name)
    return torch.cos(2.0 * torch.pi * phase)

@configclass
class ObservationsCfg:
    class PolicyCfg(ObservationGroupCfg):
        # Base velocity and orientation

        concatenate_terms: bool = True
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, history_length=5, flatten_history_dim=True)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, history_length=5, flatten_history_dim=True)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        
        # The commands we want the policy to track
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
            history_length=5, flatten_history_dim=True
        )
        
        # --- GAIT CLOCK OBSERVATIONS ---
        gait_clock_sin = ObservationTermCfg(
            func=phase_clock_sin, 
            params={"command_name": "gait_clock"},
            history_length=5, flatten_history_dim=True
        )
        gait_clock_cos = ObservationTermCfg(
            func=phase_clock_cos, 
            params={"command_name": "gait_clock"},
            history_length=5, flatten_history_dim=True
        )
        
        # Current joint state 
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel,history_length=5, flatten_history_dim=True)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel,history_length=5, flatten_history_dim=True)
        
        # Previous actions taken
        actions = ObservationTermCfg(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        
        
    policy: PolicyCfg = PolicyCfg()