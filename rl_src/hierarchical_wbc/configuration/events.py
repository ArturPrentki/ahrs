from isaaclab.managers import EventTermCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
@configclass
class EventsCfg:
    # Apply random pushes to the robot every few seconds
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        # Initialize with the base_range from the curriculum
        params={"velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}}
    )
    
    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Initialize with the base_range from the curriculum
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )