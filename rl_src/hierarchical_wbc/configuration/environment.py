from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets import UNITREE_GO2_CFG 
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import DomeLightCfg, SimulationCfg
# Import all your classes from the files above
from .action import ActionsCfg
from .commands import CommandsCfg
from .observations import ObservationsCfg
from .rewards import RewardsCfg
from .events import EventsCfg
from .terminations import TerminationsCfg
from .curriculum import CurriculumCfg

@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
    
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_GO2_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.38),  # Align spawn height closer to your 0.30 nominal base z
            joint_pos={
                "FL_hip_joint": 0.1,  "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
                "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
                "RL_hip_joint": 0.1,  "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
                "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
            }
        ),
    )
    robot.actuators["base_legs"].stiffness = 35.0
    robot.actuators["base_legs"].damping = 1.2
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3,
        track_air_time=True,
    )

@configclass
class WbcQuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the WBC quadruped environment."""

    # 1. Define Scene (Robot asset, terrain, lights)
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # (Add your robot ArticulationCfg and TerrainCfg to the scene here)
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    # 2. Basic Settings
    decimation: int = 4
    episode_length_s: float = 20.0

    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()