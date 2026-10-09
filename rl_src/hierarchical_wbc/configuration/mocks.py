import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    TerminationTermCfg,
    SceneEntityCfg
)
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

@configclass
class EmptyCfg:
    """Generic empty config for managers that don't strictly require terms."""
    pass

@configclass
class ActionsCfg:
    """Minimal GPU-native action term to satisfy RL space requirements."""
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"], 
        scale=1.0,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Minimal GPU-native observation to satisfy RL space requirements."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    """Minimal termination to prevent infinite episodes."""
    time_out = TerminationTermCfg(func=mdp.time_out)