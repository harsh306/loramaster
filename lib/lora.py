from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, dim, rank, dropout=0.1, bias=False):
        super(Encoder, self).__init__()
        self.lora_encoder = nn.Linear(dim, rank, bias=bias)
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lora_encoder(self.lora_dropout(x))


class Decoder(nn.Module):
    def __init__(self, dim, rank, bias=False):
        super(Decoder, self).__init__()
        self.lora_decoder = nn.Linear(rank, dim, bias=bias)
        nn.init.zeros_(self.lora_decoder.weight)
        if bias:
            nn.init.zeros_(self.lora_decoder.bias)

    def forward(self, x):
        return self.lora_decoder(x)

class LoRA(nn.Module):
    def __init__(self, dim, rank, dropout=0.1, bias=False):
        super(LoRA, self).__init__()
        self.encoder = Encoder(dim, rank, dropout, bias)
        self.decoder = Decoder(dim, rank, bias)

        self.lora = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def forward(self, x):
        return self.lora(x)


if __name__ == '__main__':
    # Example usage
    lora = LoRA(dim=512, rank=16, dropout=0.1)
    input_tensor = torch.randn(1, 512)
    output = lora(input_tensor)