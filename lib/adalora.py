import torch
import torch.nn as nn

class AdaLoRA(nn.Module):
    def __init__(self, dim, max_rank):
        super(AdaLoRA, self).__init__()
        self.encoder = nn.Linear(dim, max_rank)
        self.decoder = nn.Linear(max_rank, dim)
        self.importance_score = nn.Parameter(torch.ones(max_rank), requires_grad=True)

    def forward(self, x):
        x = self.encoder(x)
        x = x * self.importance_score
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    # Example usage
    adalora = AdaLoRA(dim=512, max_rank=16)
    input_tensor = torch.randn(1, 512)
    output = adalora(input_tensor)