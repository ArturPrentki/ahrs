import torch
from isaaclab.envs import ManagerBasedRLEnv

def base_height_below_threshold(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """
    Terminates the episode if the root of the robot drops below a critical Z-height.
    Acts as a virtual collision box under the robot.
    """
    # Extract the root positions in the world frame [num_envs, 3]
    root_pos = env.scene["robot"].data.root_pos_w
    
    # Return a boolean tensor where True means the Z-coordinate is too low
    return root_pos[:, 2] < minimum_height