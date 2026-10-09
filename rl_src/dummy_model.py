import torch
import torch.nn as nn
import os

class DummyTrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Ignores inputs (60D), outputs a hardcoded safe baseline (22D)
        
    def forward(self, x):
        # Batch size can be dynamic
        batch_size = x.shape[0]
        actions = torch.zeros(batch_size, 22, device=x.device)
        
        # Keep feet planted (contact states = 1.0)
        actions[:, 18:22] = 1.0
        
        # Set nominal foot positions in base frame
        # FL, FR, RL, RR (X, Y, Z)
        nominal_feet = [
            [ 0.23,  0.10, 0.0],
            [ 0.23, -0.10, 0.0],
            [-0.01,  0.10, 0.0],
            [-0.01, -0.10, 0.0]
        ]
        for i, pos in enumerate(nominal_feet):
            idx = 6 + (i * 3)
            actions[:, idx]     = pos[0] / 0.1  # divide by scale
            actions[:, idx + 1] = pos[1] / 0.1
            actions[:, idx + 2] = pos[2] / 0.1
            
        return actions

def export_dummy():
    model = DummyTrajectoryModel()
    model.eval()
    
    os.makedirs("/gardo_ws/policies", exist_ok=True)
    onnx_path = "/gardo_ws/policies/policy.onnx"
    
    dummy_input = torch.randn(1, 60, dtype=torch.float32)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['obs'], output_names=['actions'],
        dynamic_axes={'obs': {0: 'batch_size'}, 'actions': {0: 'batch_size'}}
    )
    print("Exported dummy mock policy to ONNX.")

if __name__ == "__main__":
    export_dummy()