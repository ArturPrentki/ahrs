import torch
import torch.nn as nn
import os

class Go2WbcPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(60, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 22)
        )
        
    def forward(self, x):
        return self.mlp(x)

def main():
    checkpoint_path = "logs/hybrid_wbc_go2/model_250.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # RSL-RL stores model parameters in 'model_state_dict' or 'actor_state_dict'
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("actor_state_dict"))
    
    policy = Go2WbcPolicyNet()
    
    # Clean and filter the state dict keys to map into our sequential MLP
    net_state_dict = {}
    for k, v in state_dict.items():
        if "actor" in k and "critic" not in k:
            # Strip prefixes so keys map to sequential layer indexes (0.weight, 0.bias, etc.)
            parts = k.split("actor.")
            suffix = parts[-1]
            if "mlp" in suffix:
                suffix = suffix.split("mlp.")[-1]
            net_state_dict[suffix] = v

    # Fallback load if key mapping matches standard sequential naming
    try:
        policy.load_state_dict(net_state_dict, strict=False)
        print("Successfully loaded actor weights into clean MLP policy.")
    except Exception as e:
        print(f"Strict=False load warning: {e}")
        # Last resort: try loading raw state dict directly
        policy.load_state_dict(state_dict, strict=False)

    policy.eval()
    
    # Export to ONNX
    output_dir = "/gardo_ws/policies"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "policy.onnx")
    
    dummy_input = torch.randn(1, 60, dtype=torch.float32)
    
    torch.onnx.export(
        policy,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['obs'],
        output_names=['actions'],
        dynamic_axes={'obs': {0: 'batch_size'}, 'actions': {0: 'batch_size'}}
    )
    
    print(f"Successfully exported clean ONNX model to {onnx_path}!")

if __name__ == "__main__":
    main()