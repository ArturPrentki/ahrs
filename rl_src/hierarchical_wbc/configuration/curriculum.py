import isaaclab.envs.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel
from isaaclab.managers import CurriculumTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.utils import configclass
# ---------------------------------------------------------
# Custom Curriculum Functions
# ---------------------------------------------------------
def modify_command_ranges(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    new_range: tuple, 
    num_steps: int
):
    if env.common_step_counter > num_steps:
        command_term = env.command_manager.get_term(term_name)
        
        # FIX: Modify the active instance attributes, not just the cfg
        # Check the exact attribute names in Isaac Lab's UniformVelocityCommand
        if hasattr(command_term, "lin_vel_x_range"):
            command_term.lin_vel_x_range = new_range
        
        # It's good practice to update the cfg too, just for logging consistency
        command_term.cfg.ranges.lin_vel_x = new_range
def scale_push_velocity(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    base_range: tuple,
    max_range: tuple,
    max_steps: int
):
    progress = min(1.0, env.common_step_counter / max_steps)
    current_min = base_range[0] + progress * (max_range[0] - base_range[0])
    current_max = base_range[1] + progress * (max_range[1] - base_range[1])
    
    # FIX: Access the event term's kwargs directly through the active events dictionary
    # Depending on your Isaac Lab version, this is usually stored in env.event_manager._term_cfgs
    # or env.event_manager.cfg
    
    # Example for recent Isaac Lab versions:
    if hasattr(env.event_manager, "cfg") and hasattr(env.event_manager.cfg, term_name):
        term_cfg = getattr(env.event_manager.cfg, term_name)
        term_cfg.params["velocity_range"]["x"] = (current_min, current_max)
        term_cfg.params["velocity_range"]["y"] = (current_min, current_max)

def scale_friction_randomization(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    base_range: tuple,
    max_range: tuple,
    max_steps: int
):
    """Linearly widens the friction range from base to max over max_steps."""
    progress = min(1.0, env.common_step_counter / max_steps)
    
    current_min = base_range[0] + progress * (max_range[0] - base_range[0])
    current_max = base_range[1] + progress * (max_range[1] - base_range[1])
    
    event_cfg = env.event_manager.get_term_cfg(term_name)
    event_cfg.params["static_friction_range"] = (current_min, current_max)
    event_cfg.params["dynamic_friction_range"] = (current_min, current_max)


# ---------------------------------------------------------
# Configuration Class
# ---------------------------------------------------------

@configclass
class CurriculumCfg:
    """Curriculum terms for the environment."""

    # 1. Delayed Penalties:
    # Start with 0 penalty on joint velocities to encourage exploration.
    # After 5,000 steps, turn on the penalty to smooth out the task-space commands.
    # penalize_joint_vel = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "dof_vel_l2",  # Must perfectly match the name in WbcRewardsCfg
    #         "weight": -1e-4,
    #         "num_steps": 5000, 
    #     }
    # )

    penalize_action_rate = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rate_l2", # Smooth out task-space targets sent to OSQP
            "weight": -0.01,
            "num_steps": 10000,
        }
    )
# Slowly increase the push velocity from tiny nudges to full shoves
    increase_push_intensity = CurriculumTermCfg(
        func=scale_push_velocity,
        params={
            "term_name": "push_robot", # Must match the name in WbcEventsCfg
            "base_range": (-0.05, 0.05), # Start with tiny pushes
            "max_range": (-0.6, 0.6),    # Scale up to strong pushes
            "max_steps": 5000,           # Policy steps to reach max intensity
        }
    )

    # Slowly widen the friction randomization from nearly uniform to highly variable
    increase_friction_variance = CurriculumTermCfg(
        func=scale_friction_randomization,
        params={
            "term_name": "randomize_friction", 
            "base_range": (0.8, 1.0),    # Start with predictable, high friction
            "max_range": (0.3, 1.2),     # Scale out to slippery and sticky surfaces
            "max_steps": 5000,
        }
    )
    # Increase the maximum forward velocity target after 20,000 steps.
    increase_x_vel = CurriculumTermCfg(
        func=modify_command_ranges,
        params={
            "term_name": "base_velocity", 
            "new_range": (-1.0, 1.5), # Expand the velocity range
            "num_steps": 20000,
        }
    )
    
    # # 3. Terrain Progression (If using a stepped terrain curriculum)
    # # Automatically moves the robot to harder terrain levels if tracking error is low
    # terrain_levels = CurriculumTermCfg(
    #     func=terrain_levels_vel,
    #     params={"err_threshold": 0.1},
    # )