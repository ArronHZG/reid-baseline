from torch import nn

from utils import Data


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, code_dim, bias=True):
        super(AutoEncoder, self).__init__()

        times = int(input_dim / code_dim)

        self.encoder = nn.Sequential(
            # fc layers for the encoder
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, int(input_dim / 2), bias=bias),
            nn.LeakyReLU(),

            nn.BatchNorm1d(int(input_dim / 2)),
            nn.Linear(int(input_dim / 2), code_dim, bias=bias),
            nn.Sigmoid(),

        )

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(code_dim),
            nn.Linear(code_dim, int(input_dim / 2), bias=bias),
            nn.LeakyReLU(),
            nn.BatchNorm1d(int(input_dim / 2)),
            nn.Linear(int(input_dim / 2), input_dim, bias=bias),
        )
        # fc layers for the decoder

    def forward(self, x):
        data = Data()
        code = self.encoder(x)
        recon = self.decoder(code)
        data.recon_ae = recon
        return data


if __name__ == '__main__':
    import torch

    input = torch.randn(2, 4)
    model = AutoEncoder(4, 2)
    r = model(input)
