import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse

def phase_clock_observation(env: ManagerBasedRLEnv, action_term_name: str) -> torch.Tensor:
    """Computes periodic phase clock signals (sin, cos) from the WBC action term.
    
    Args:
        env: The environment instance.
        action_term_name: Name of the action term managing the phase clock (e.g., 'wbc_command').
        
    Returns:
        Tensor of shape (num_envs, 2) containing [sin(2*pi*phase), cos(2*pi*phase)].
    """
    # 1. Retrieve the active WBC action term instance from the action manager
    action_term = env.action_manager.get_term(action_term_name)
    
    # 2. Extract phase scalar (shape: (num_envs,))
    phase = action_term.phase
    
    # 3. Map normalized phase [0, 1) to radians [0, 2*pi)
    phase_rad = phase * (2.0 * torch.pi)
    
    # 4. Return sin and cos stacked along the last dimension (shape: (num_envs, 2))
    return torch.stack([torch.sin(phase_rad), torch.cos(phase_rad)], dim=-1)


def foot_positions_in_base_frame(env: ManagerBasedRLEnv, asset_name: str = "robot") -> torch.Tensor:
    """
    Computes the 3D Cartesian positions of the 4 feet relative to the robot's base frame.
    
    Returns:
        Tensor of shape (num_envs, 12) containing [x, y, z] for FL, FR, RL, RR.
    """
    robot = env.scene[asset_name]
    
    # 1. Get foot body indices
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    try:
        foot_indices = [robot.find_bodies(name)[0][0] for name in foot_names]
    except Exception:
        foot_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        foot_indices = [robot.find_bodies(name)[0][0] for name in foot_names]
        
    # 2. Extract from guaranteed PyTorch state tensors instead of isolated properties
    # root_state_w shape is [num_envs, 13] -> [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    root_pos_w = robot.data.root_state_w[:, 0:3] 
    root_quat_w = robot.data.root_state_w[:, 3:7] 
    
    # body_state_w shape is [num_envs, num_bodies, 13]
    foot_pos_w = robot.data.body_state_w[:, foot_indices, 0:3] 
    
    # 3. Transform to the local base frame
    num_envs = env.num_envs
    local_foot_pos = torch.zeros((num_envs, 4, 3), device=env.device)
    
    for i in range(4):
        # Find relative translation in the world frame
        rel_pos_w = foot_pos_w[:, i, :] - root_pos_w
        # Rotate the relative vector into the base frame
        local_foot_pos[:, i, :] = quat_apply_inverse(root_quat_w, rel_pos_w)
        
    # 4. Flatten to (num_envs, 12)
    return local_foot_pos.view(num_envs, -1)


def physical_contact_states(env: ManagerBasedRLEnv, sensor_name: str = "contact_forces", threshold: float = 10.0) -> torch.Tensor:
    """
    Computes a binary tensor indicating if each foot is currently touching the ground.
    
    Returns:
        Tensor of shape (num_envs, 4) containing 1.0 for contact and 0.0 for swing.
    """
    contact_sensor = env.scene[sensor_name]
    
    # 1. Get foot body indices from the contact sensor
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    try:
        foot_indices = [contact_sensor.find_bodies(name)[0][0] for name in foot_names]
    except Exception:
        foot_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        foot_indices = [contact_sensor.find_bodies(name)[0][0] for name in foot_names]
        
    # 2. Get the vertical (Z-axis) contact forces
    net_forces_z = contact_sensor.data.net_forces_w[:, foot_indices, 2] 
    
    # 3. Apply threshold to filter out simulation noise and cast to float
    is_contact = (net_forces_z > threshold).float()
    
    return is_contact

def cpg_phase_sin_cos(env, action_term_name: str):
    phase = env.action_manager.action_terms[action_term_name].phase
    return torch.stack([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)