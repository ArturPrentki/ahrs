
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
def trot_gait_reward(env, action_term_name: str):
    action_term = env.action_manager.get_term(action_term_name)
    intended_contacts = action_term.contact_targets  # [N, 4]
    
    # Get actual contacts from sensor
    contact_sensor = env.scene.sensors["contact_forces"]
    foot_indices, _ = contact_sensor.find_bodies(".*_foot")
    actual_forces = contact_sensor.data.net_forces_w[:, foot_indices, :]  
    actual_contacts = (torch.norm(actual_forces, dim=-1) > 1.0).float()  # [N, 4]
    
    # Reward for matching the CPG state (1.0 if match, 0.0 if mismatch)
    match = (intended_contacts == actual_contacts).float()
    return torch.mean(match, dim=-1)

def contact_target_penalty(env:ManagerBasedRLEnv, action_term_name: str) -> torch.Tensor:

    """Penalizes deviations between the policy's predicted contact flags (actions 18-22)
    and the analytical trot gait phase schedule.
    
    Trot Gait Schedule:
      - Phase in [0.0, 0.5): Stance = [FL, RR] (1.0), Swing = [FR, RL] (0.0)
      - Phase in [0.5, 1.0): Stance = [FR, RL] (1.0), Swing = [FL, RR] (0.0)
    """
    # 1. Access the WBC action term instance
    action_term = env.action_manager.get_term(action_term_name)
    
    # 2. Extract policy contact output (Actions 18-22, shape: (num_envs, 4))
    # Using raw_actions or contact_targets depending on your process_actions assignment
    policy_contact_targets = action_term._raw_actions[:, 18:22]
    
    # 3. Compute analytical desired contact flags based on current phase
    phase = action_term.phase  # shape: (num_envs,)
    
    group1_stance = (phase < 0.5).float()  # FL, RR active
    group2_stance = (phase >= 0.5).float()  # FR, RL active
    
    # Order: [FL, FR, RL, RR] to match your ROS 2 FootPositionTarget[4] array
    desired_contacts = torch.stack([
        group1_stance,  # FL
        group2_stance,  # FR
        group2_stance,  # RL
        group1_stance   # RR
    ], dim=-1)
    
    # 4. Compute MSE loss across the 4 foot contact flags per environment
    # Shape returned: (num_envs,)
    mse_error = torch.mean(torch.square(policy_contact_targets - desired_contacts), dim=1)
    
    return mse_error

def action_vel_target_penalty(env: ManagerBasedRLEnv, action_term_name: str, command_name: str) -> torch.Tensor:
    """
    Penalizes the policy if its commanded base velocity (Actions 0-2) 
    deviates from the user's joystick command.
    """
    action_term = env.action_manager.get_term(action_term_name)
    # The scaled output from the network (num_envs, 3)
    policy_vel_targets = action_term.base_vel_targets 
    
    # The actual joystick command (num_envs, 3) -> [vx, vy, yaw_rate]
    joystick_command = env.command_manager.get_command(command_name)
    
    # Calculate Mean Squared Error between what the policy asked for and the joystick
    mse_error = torch.mean(torch.square(policy_vel_targets - joystick_command[:, :3]), dim=1)
    
    return mse_error

def wbc_foot_clearance_penalty(env, action_term_name: str, min_lateral_dist: float, min_longitudinal_dist: float):
    action_term = env.action_manager.get_term(action_term_name)
    targets = action_term.foot_pos_targets # [N, 4, 3] (Local frame targets)
    
    # 0=FL, 1=FR, 2=RL, 3=RR
    # Lateral distance between FL(0) and FR(1), RL(2) and RR(3)
    front_lat = targets[:, 0, 1] - targets[:, 1, 1]
    rear_lat = targets[:, 2, 1] - targets[:, 3, 1]
    
    lateral_violation = torch.relu(min_lateral_dist - front_lat) + torch.relu(min_lateral_dist - rear_lat)
    
    return lateral_violation # Returns > 0 if feet are too close

def contact_gait_tracking_reward(env, action_term_name: str) -> torch.Tensor:
    """Penalizes deviation between policy contact outputs and reference trot phase."""
    action_term = env.action_manager.get_term(action_term_name)
    
    # Policy's contact probabilities (num_envs, 4)
    policy_contact_probs = action_term.contact_probs 
    
    # Generate reference contact states from phase clock (3 Hz trot)
    phase = action_term.phase  # (num_envs,)
    
    # Trot Gait: FL(0) and RR(3) in stance when phase < 0.5
    #           FR(1) and RL(2) in stance when phase >= 0.5
    ref_stance_fl_rr = (phase < 0.5).float().unsqueeze(-1)
    ref_stance_fr_rl = (phase >= 0.5).float().unsqueeze(-1)
    
    ref_contacts = torch.cat([
        ref_stance_fl_rr,  # FL
        ref_stance_fr_rl,  # FR
        ref_stance_fr_rl,  # RL
        ref_stance_fl_rr   # RR
    ], dim=-1)
    
    # Mean Squared Error penalty between predicted contact probability and target phase
    error = torch.mean(torch.square(policy_contact_probs - ref_contacts), dim=-1)
    return -error

def foot_slip_penalty(env, sensor_cfg, asset_cfg):
    # Extract foot velocities from the asset
    robot = env.scene[asset_cfg.name]
    foot_indices = robot.find_bodies(asset_cfg.body_names)[0]
    foot_vel = robot.data.body_lin_vel_w[:, foot_indices, :2] # [N, 4, 2] (XY vel)
    
    # Extract actual contacts
    contact_sensor = env.scene[sensor_cfg.name]
    foot_indices, _ = contact_sensor.find_bodies(".*_foot")
    forces = contact_sensor.data.net_forces_w[:, foot_indices, :]  
    in_contact = (torch.norm(forces, dim=-1) > 1.0).float() # [N, 4]
    
    # Penalize squared XY velocity only when in contact
    slip_penalty = torch.sum(in_contact * torch.sum(torch.square(foot_vel), dim=-1), dim=-1)
    return slip_penalty