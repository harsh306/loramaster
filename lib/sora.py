import torch
import torch.nn as nn
import torch.nn.functional as F

class SoRA(nn.Module):
    def __init__(self, dim, rank, sparse_ratio):
        super(SoRA, self).__init__()
        self.encoder = nn.Linear(dim, rank)
        self.decoder = nn.Linear(rank, dim)
        self.sparse_gate = nn.Linear(rank, rank)
        self.sparse_ratio = sparse_ratio

    def forward(self, x):
        x = self.encoder(x)
        gate = torch.sigmoid(self.sparse_gate(x))
        sparse_mask = (gate > self.sparse_ratio).float()
        x = x * sparse_mask
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    # Example usage
    sora = SoRA(dim=512, rank=16, sparse_ratio=0.5)
    input_tensor = torch.randn(1, 512)
    output = sora(input_tensor)