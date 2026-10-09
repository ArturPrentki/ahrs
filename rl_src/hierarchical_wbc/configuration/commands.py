import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class GaitClockCommand(CommandTerm):
    """Generates a continuous [0, 1) phase clock for 4 legs to dictate swing/stance."""
    cfg: "GaitClockCommandCfg"

    def __init__(self, cfg: "GaitClockCommandCfg", env):
        super().__init__(cfg, env)
        self.freq = self.cfg.gait_frequency
        
        # FIX: Assign tensor directly to self instead of register_buffer
        self.phase_offsets = torch.tensor(self.cfg.phase_offsets, device=self.device)
        
        self._master_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._clock = torch.zeros(self.num_envs, 4, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._resample_command(env_ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        # Randomize phase ONLY for resetting envs
        self._master_phase[env_ids] = torch.rand(len(env_ids), 1, device=self.device)
        self._clock[env_ids] = torch.fmod(self._master_phase[env_ids] + self.phase_offsets, 1.0)

    def _update_command(self):
        # Step phase forward for all envs during normal step
        self._master_phase += self.freq * self._env.step_dt
        self._master_phase = torch.fmod(self._master_phase, 1.0)
        self._clock = torch.fmod(self._master_phase + self.phase_offsets, 1.0)

    def _update_metrics(self):
        self.metrics["gait_frequency"] = torch.full((self.num_envs,), self.freq, device=self.device)
        self.metrics["mean_phase_progress"] = torch.mean(self._clock, dim=-1)

    @property
    def command(self) -> torch.Tensor:
        """Returns cyclic sine/cosine signals (shape: num_envs, 8) for smooth observations."""
        sin_clock = torch.sin(2 * torch.pi * self._clock)
        cos_clock = torch.cos(2 * torch.pi * self._clock)
        return torch.cat([sin_clock, cos_clock], dim=-1)

# Link the class to the config

@configclass
class GaitClockCommandCfg(CommandTermCfg):
    """Configuration for the phase clock command generator."""
    class_type: GaitClockCommand = GaitClockCommand
       
    gait_frequency: float = 2.0  # Speed of the step cycle in Hz
    # Offsets for FL, FR, RL, RR (0.0 to 1.0)
    phase_offsets: tuple = (0.0, 0.5, 0.5, 0.0) 
    
    resampling_time_range: tuple[float, float] = (4.0, 10.0)


@configclass
class CommandsCfg:
    # Generates random x, y, and yaw velocity targets
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

    gait_clock: GaitClockCommandCfg = GaitClockCommandCfg(
        gait_frequency=2.0,                  # 2 step cycles per second
        phase_offsets=(0.0, 0.5, 0.5, 0.0)   # Trot pattern

    )