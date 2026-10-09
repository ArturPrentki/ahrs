import argparse
import torch
import numpy as np
# from isaaclab.utils import configclass

# 1. Isaac Lab Application Launcher (Must be first)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a Go2 hybrid WBC policy.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
# from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg


# from isaaclab.managers import RewardTermCfg as RewTerm
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from environment_config import Go2WbcEnvCfg, Go2WbcEnv

def main():
    # 1. Instantiate explicitly
    env_cfg = Go2WbcEnvCfg()
    
    # Apply CLI args
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 2. Initialize Environment
    env = Go2WbcEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)
    
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

    runner = OnPolicyRunner(wrapped_env, agent_cfg, log_dir="logs/hybrid_wbc_go2")
    # print(env.unwrapped.scene["robot"].joint_names)
    
    runner.learn(num_learning_iterations=1500, init_at_random_ep_len=True)
    wrapped_env.close()
    simulation_app.close()
if __name__ == "__main__":
    main()