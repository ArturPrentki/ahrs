import torch

class BatchedQuadrupedIK:
    """Fast, loop-free analytical IK for standard 3-DOF quadruped legs."""
    def __init__(self, l1=0.0955, l2=0.213, l3=0.213, device='cuda'):
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.device = device
        
        self.max_reach = self.l2 + self.l3 - 1e-4
        self.min_reach = abs(self.l2 - self.l3) + 1e-4
        
        # Signs for Y-offset: FL(+), FR(-), RL(+), RR(-)
        self.side_signs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device).view(1, 4)

    def solve(self, foot_pos_hip_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            foot_pos_hip_frame: Tensor of shape (num_envs, 4, 3) in [x, y, z] per hip origin.
        Returns:
            Tensor of shape (num_envs, 12) ordered [FL_q0..q2, FR_q0..q2, RL_q0..q2, RR_q0..q2]
        """
        # side_signs = self.side_signs.to(foot_pos_hip_frame.device)
        
        # Clone to safely adjust rear hip frames for Go2 URDF conventions
        
        x = foot_pos_hip_frame[..., 0]
        y = foot_pos_hip_frame[..., 1]
        z = foot_pos_hip_frame[..., 2]
        
        # Hip Roll (q0)
        L_yz = torch.sqrt(y**2 + z**2)
        L_yz_clamped = torch.clamp(L_yz, min=self.l1 + 1e-5)
        alpha = torch.atan2(-y, -z)
        
        offset_ratio = torch.clamp(self.side_signs * self.l1 / L_yz_clamped, -1.0 + 1e-6, 1.0 - 1e-6)
        offset_angle = torch.asin(offset_ratio)
        q0 = alpha + offset_angle
        
        # Sagittal Distance (L_xz)
        z_prime = -torch.sqrt(torch.clamp(L_yz_clamped**2 - self.l1**2, min=1e-8))
        L_xz = torch.sqrt(x**2 + z_prime**2)
        
        L_xz_clamped = torch.clamp(L_xz, min=self.min_reach, max=self.max_reach)
        
        # Knee Pitch (q2)
        cos_q2 = (self.l2**2 + self.l3**2 - L_xz_clamped**2) / (2 * self.l2 * self.l3)
        cos_q2 = torch.clamp(cos_q2, -1.0 + 1e-6, 1.0 - 1e-6)
        q2 = -(torch.pi - torch.acos(cos_q2))
        
        # Hip Pitch (q1)
        theta = torch.atan2(x, -z_prime)
        cos_beta = (self.l2**2 + L_xz_clamped**2 - self.l3**2) / (2 * self.l2 * L_xz_clamped)
        cos_beta = torch.clamp(cos_beta, -1.0 + 1e-6, 1.0 - 1e-6)
        beta = torch.acos(cos_beta)
        q1 = theta + beta
        
        return torch.stack([q0, q1, q2], dim=-1).view(-1, 12)