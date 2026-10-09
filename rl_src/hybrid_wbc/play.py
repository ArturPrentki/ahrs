import argparse
import torch
import os
import gymnasium as gym

from isaaclab.app import AppLauncher

# 1. Launch Isaac Sim app first
parser = argparse.ArgumentParser(description="Play a trained Go2 hybrid WBC policy.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--model_id", type=int, default=1499, help="Id of model to play.")

args_cli = parser.parse_args()
# Force cameras to be enabled so the viewport can render to an RGB array
args_cli.enable_cameras = True 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports after app launch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_config import Go2WbcEnvCfg

# Pass the parsed arguments into the main function
def main(args):
    # Load configuration
    env_cfg = Go2WbcEnvCfg()

    # 1. FORCE UPRIGHT BASE ORIENTATION & HEIGHT
    # Quaternion order in Isaac Lab is (w, x, y, z). (1, 0, 0, 0) is perfectly upright.
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.38)
    env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0) 
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)

    # 2. FORCE NOMINAL STANDING JOINT POSITIONS
    # Overrides joint reset variations with the default asset joint configuration
    if hasattr(env_cfg.scene.robot.init_state, "default_joint_angles"):
        env_cfg.scene.robot.init_state.joint_pos = env_cfg.scene.robot.init_state.default_joint_angles

    # 3. STRIP RESET DYNAMICS & NOISE
    if hasattr(env_cfg, "events"):
        env_cfg.events.reset_robot_joints = None
        env_cfg.events.reset_joints_by_scale = None
        env_cfg.events.reset_base = None

    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False

    # Keep your custom velocity overrides
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
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
        # Add a prefix so different model IDs save as different files
        name_prefix=f"rl-video-model_{args.model_id}" 
    )
    # ------------------------------------------------

    wrapped_env = RslRlVecEnvWrapper(env)
    
    # Use the model_id argument to construct the file path dynamically
    checkpoint_path = f"/gardo_ws/logs/hybrid_wbc_go2/model_{args.model_id}.pt" 
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # EXACT agent_cfg dictionary used during training
    agent_cfg = {
        "seed": 1,
    
        "save_interval": 50,
        "runner": {
            "run_name": "go2_wbc",
            "experiment_name": "hybrid_wbc_go2",
            "max_iterations": 1500,
            "resume": False,
        },
        "num_steps_per_env": 18, 
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
            "desired_kl": 0.008,
            "max_grad_norm": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 512, 256],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.20, "std_type": "scalar"}
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 512, 256],
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
            
            # Optional: Print progress
            if step % 50 == 0:
                print(f"Recorded {step}/400 steps...")

    print(f"Video saved to {video_dir}!")
    wrapped_env.close()
    simulation_app.close()

if __name__ == "__main__":
    # Pass the parsed arguments into main
    main(args_cli)