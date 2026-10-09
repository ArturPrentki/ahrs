import argparse
import torch

# 1. Isaac Lab App Launcher (Must be imported and initialized FIRST)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a Go2 hybrid WBC policy.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports must happen AFTER simulation_app is created
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import your master config
from configuration.environment import WbcQuadrupedEnvCfg 

def main():
    # 1. Instantiate the Configuration
    env_cfg = WbcQuadrupedEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 2. Initialize the Base Environment
    # We use ManagerBasedRLEnv directly, passing your custom config.
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 3. Wrap it for RSL-RL
    wrapped_env = RslRlVecEnvWrapper(env)
    
    # 4. Define RL Agent Parameters (PPO)
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
    # 5. Initialize Runner and Start Training
    runner = OnPolicyRunner(wrapped_env,
                            agent_cfg,
                            log_dir="logs/hierarhical_wbc_go2",
                            device=env.unwrapped.device)
    
    print("[INFO] Starting PPO Training Loop...")
    # torch.cuda.set_sync_debug_mode("error")
    runner.learn(num_learning_iterations=1500, init_at_random_ep_len=True)
    
    # Clean up securely
    wrapped_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()