import argparse
import os
import torch

# 1. Isaac Lab App Launcher (Must be imported and initialized FIRST)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Zero-Velocity Trot for Go2 WBC policy.")
parser.add_argument("--log_dir", type=str, default="logs/hierarhical_wbc_go2", help="Path to the log directory.")
parser.add_argument("--model", type=str, default="model_1499.pt", help="Name of the model file to load.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports must happen AFTER simulation_app is created
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import your master config (adjust the import path as needed)
from configuration.environment import WbcQuadrupedEnvCfg 

def main():
    # 3. Instantiate the Configuration and Override for Zero-Velocity
    env_cfg = WbcQuadrupedEnvCfg()
    env_cfg.scene.num_envs = 2  # Small number of environments for visualization
    
    # Force velocity commands to exactly 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    # 4. Initialize the Environment and RSL-RL Wrapper
    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)
    
    # 5. Define RL Agent Parameters (Minimal config required for inference)
    agent_cfg = {
            "seed": 1,
            "save_interval": 50,
            "runner": {
                "run_name": "go2_hierarhical_wbc",
                "experiment_name": "hierarhical_wbc_go2",
                "max_iterations": 1500,
                "resume": False,
            },
            "num_steps_per_env": 48, # Standard for quadruped locomotion
            "multi_gpu": None,
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "algorithm": {
                "class_name": "rsl_rl.algorithms.ppo:PPO",
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 1e-4,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [512, 256, 128],
                "activation": "elu",
                "obs_normalization": True, # Crucial for task-space inputs
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.40, "std_type": "scalar"}
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [512, 256, 128],
                "activation": "elu",
                "obs_normalization": True,
            }
        }
    
    # 6. Locate the Latest Checkpoint
    if not os.path.exists(args_cli.log_dir):
        raise FileNotFoundError(f"Log directory not found: {args_cli.log_dir}")
        
    latest_run_dir = args_cli.log_dir 
    resume_path = os.path.join(latest_run_dir, args_cli.model)
    print(f"[INFO] Loading model from: {resume_path}")
    
    # 7. Load Runner and Extract Policy
    runner = OnPolicyRunner(wrapped_env, agent_cfg, log_dir=latest_run_dir, device=env.unwrapped.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # 8. Inference Loop
    obs, _ = wrapped_env.reset()
    print("[INFO] Starting Zero-Velocity Trot-in-Place Test...")
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped_env.step(actions)
            
            # Access action targets
            task_space_action = env.unwrapped.action_manager.get_term("task_space")
            foot_targets = task_space_action.target_foot_positions_ground_frame
            
            z_heights = foot_targets[0, :, 2]
            x_targets = foot_targets[0, :, 0]
            y_targets = foot_targets[0, :, 1]
            
            # --- Infer Contact States from Gait Clock Phase ---
            gait_term = env.unwrapped.command_manager.get_term("gait_clock")
            # _clock holds phase values [0, 1) for [FL, FR, RL, RR] in env 0
            clock_vals = gait_term._clock[0]
            # Stance phase is active when clock <= 0.5
            inferred_contacts = (clock_vals <= 0.5).int()
            #|  FL  |   FR  |   RL  |   RR  |  
            #| X.XXX | X.XXX | X.XXX | X.XXX |
            #| Y.YYY | Y.YYY | Y.YYY | Y.YYY |
            #| Z.ZZZ | Z.ZZZ | Z.ZZZ | Z.ZZZ |
            print(f"| FL {inferred_contacts[0].item()}  | FR {inferred_contacts[1].item()}  | RL {inferred_contacts[2].item()}  | RR {inferred_contacts[3].item()}  |")
            print(f"| {x_targets[0]:2.3f} | {x_targets[1]:2.3f} | {x_targets[2]:2.3f} | {x_targets[3]:2.3f} |")
            print(f"| {y_targets[0]:2.3f} | {y_targets[1]:2.3f} | {y_targets[2]:2.3f} | {y_targets[3]:2.3f} |")
            print(f"| {z_heights[0]:2.3f} | {z_heights[1]:2.3f} | {z_heights[2]:2.3f} | {z_heights[3]:2.3f} |")
            print("-----------------------------------------------------")
    # Clean up
    wrapped_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()