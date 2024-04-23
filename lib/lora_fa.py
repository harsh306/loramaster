import torch
import torch.nn as nn

class LoRA_FA(nn.Module):
    def __init__(self, dim, rank):
        super(LoRA_FA, self).__init__()
        self.A = nn.Linear(dim, dim)  # Frozen pre-trained weights
        self.B = nn.Linear(dim, rank)  # Learnable low-rank decomposition

        # Freeze the pre-trained weights
        self.A.weight.requires_grad = False
        self.A.bias.requires_grad = False

    def forward(self, x):
        x = self.A(x)  # Frozen pre-trained weights
        x = self.B(x)  # Learnable low-rank decomposition
        return x

if __name__ == '__main__':
    # Example usage
    lora_fa = LoRA_FA(dim=512, rank=16)
    input_tensor = torch.randn(1, 512)
    output = lora_fa(input_tensor)