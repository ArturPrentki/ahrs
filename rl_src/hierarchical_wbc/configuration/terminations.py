from isaaclab.managers import TerminationTermCfg, SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    illegal_contact = TerminationTermCfg(
        func=mdp.illegal_contact, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 1.0})

    # base_contact = TerminationTermCfg(
    #     func=mdp.illegal_contact, params={"sensor_cfg": ContactSensorCfg(body_names=["base"])}
    # )