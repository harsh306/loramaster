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


class EncoderVector(nn.Module):
    def __init__(self, rank):
        super(EncoderVector, self).__init__()
        self.lora_vector = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        return x * self.lora_vector

class DecoderVector(nn.Module):
    def __init__(self, rank):
        super(DecoderVector, self).__init__()
        self.lora_vector = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        return x * self.lora_vector

class Vera(nn.Module):
    def __init__(self, dim, rank, dropout=0.1, bias=False):
        super(Vera, self).__init__()
        self.encoder = Encoder(dim, rank, dropout, bias)
        self.decoder = Decoder(dim, rank, bias)
        self.encoder_vector = EncoderVector(rank)
        self.decoder_vector = DecoderVector(rank)

        self.vera = nn.Sequential(
            self.encoder,
            self.encoder_vector,
            self.decoder,
            self.decoder_vector)

    def forward(self, x):
        return self.vera(x)

if __name__ == '__main__':
    # Example usage
    vera = Vera(dim=512, rank=16, dropout=0.1)
    input_tensor = torch.randn(1, 512)
    output = vera(input_tensor)