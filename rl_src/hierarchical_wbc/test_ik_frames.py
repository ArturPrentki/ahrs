import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Go2 IK Frame Orientation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False
args_cli.livestream = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import UNITREE_GO2_CFG
from kinematics import BatchedQuadrupedIK
import torch

# ---------------------------------------------------------
# Minimal Scene Setup
# ---------------------------------------------------------
@configclass
class TestSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/Robot")
    # Increase stiffness so the robot rigidly holds the IK targets
    robot.actuators["base_legs"].stiffness = 100.0
    robot.actuators["base_legs"].damping = 2.0
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(TestSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    
    # Suspend the robot in the air slightly so we can see the leg posture clearly
    root_pose = scene["robot"].data.default_root_pose.torch.clone()
    root_pose[:, 2] = 0.45 
    scene["robot"].write_root_link_pose_to_sim_index(root_pose=root_pose)
    
    root_vel = scene["robot"].data.default_root_vel.torch.clone()
    scene["robot"].write_root_com_velocity_to_sim_index(root_velocity=root_vel)
    
    ik_solver = BatchedQuadrupedIK(device=sim.device)
    
    # Target: Straight down 0.3m for all 4 legs [num_envs, 4, 3]
    foot_targets = torch.tensor([[
        [0.0, 0.0, -0.3], # FL
        [0.0, 0.0, -0.3], # FR
        [0.0, 0.0, -0.3], # RL
        [0.0, 0.0, -0.3]  # RR
    ]], device=sim.device)

    print("[INFO] Applying static IK targets. Check the viewer.")
    
    while simulation_app.is_running():
        # 1. Solve IK
        joint_pos_targets = ik_solver.solve(foot_targets)
        
        # 2. Apply to robot using the updated indexed target method
        scene["robot"].set_joint_position_target_index(target=joint_pos_targets)
        
        # 3. Step simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

if __name__ == "__main__":
    main()
    simulation_app.close()