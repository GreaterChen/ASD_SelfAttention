from requirements import *
from args import *
from utils import *
import torch.utils.data


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)


class StackAutoEncoder(nn.Module):
    def __init__(self, encoders):
        super(StackAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(encoders[0], encoders[1])
        self.decoder = nn.Sequential(encoders[2], encoders[3])

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out
