import argparse
import torch
import os
import gymnasium as gym

# 1. Launch Isaac Sim app first
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Go2 hierarchical WBC policy.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--model_id", type=int, default=1499, help="Id of model to play.")

args_cli = parser.parse_args()
# Force cameras to be enabled so the viewport can render to an RGB array for video
args_cli.enable_cameras = True 
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports after app launch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import the NEW master config
from configuration.environment import WbcQuadrupedEnvCfg 

def main(args):
    # Load the new configuration
    env_cfg = WbcQuadrupedEnvCfg()

    # 1. FORCE UPRIGHT BASE ORIENTATION & HEIGHT
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.38)
    # using new quaternion ordering xyzw
    env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0) 
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)

    # 2. STRIP RESET DYNAMICS & NOISE
    # if hasattr(env_cfg, "events"):
    #     # Disable domain randomization / pushes during evaluation
    #     env_cfg.events.push_robot = None
    #     env_cfg.events.reset_base = None
    #     env_cfg.events.reset_robot_joints = None

    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False

    # Command fixed forward velocity for the evaluation video
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0) 
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    
    # --- Wrap environment for video recording ---
    video_dir = os.path.join(os.getcwd(), "videos")
    print(f"Recording video to: {video_dir}")
    
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=video_dir, 
        step_trigger=lambda step: step == 0,
        video_length=200,
        disable_logger=True,
        name_prefix=f"rl-video-model_{args.model_id}" 
    )
    # ------------------------------------------------

    wrapped_env = RslRlVecEnvWrapper(env)
    
    # Point to the NEW logging directory from train.py
    log_dir = os.path.join(os.getcwd(), "logs", "hierarhical_wbc_go2")
    checkpoint_path = os.path.join(log_dir, f"model_{args.model_id}.pt")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # EXACT agent_cfg dictionary used in the new train.py
    agent_cfg = {
        "seed": 1,
        "save_interval": 50,
        "runner": {
            "run_name": "go2_hierarhical_wbc",
            "experiment_name": "hierarhical_wbc_go2",
            "max_iterations": 1500,
            "resume": False,
        },
        "num_steps_per_env": 48, 
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
            "hidden_dims": [512, 256, 128], # Updated to match train.py
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.20, "std_type": "scalar"}
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128], # Updated to match train.py
            "activation": "elu",
            "obs_normalization": True,
        }
    }
    
    # Initialize the runner and load weights
    runner = OnPolicyRunner(wrapped_env, agent_cfg, log_dir=None, device="cuda:0")
    runner.load(checkpoint_path)
    
    # Extract policy model for evaluation
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy.eval()
    
    obs, _ = wrapped_env.reset()
    
    print("Running policy evaluation and recording video...")
    with torch.no_grad():
        for step in range(400):
            if not simulation_app.is_running():
                break
                
            actions = policy(obs)
            obs, reward, dones, extras = wrapped_env.step(actions)
            
            if step % 50 == 0:
                print(f"Recorded {step}/400 steps...")

    print(f"Video saved to {video_dir}!")
    wrapped_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main(args_cli)