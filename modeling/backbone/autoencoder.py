from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, code_dim, bias=False):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # fc layers for the encoder
            nn.Linear(input_dim, code_dim, bias=bias),
            nn.LeakyReLU())

        # fc layers for the decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, input_dim, bias=bias),
            # nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon


if __name__ == '__main__':
    import torch

    input = torch.randn(2, 4)
    model = AutoEncoder(4, 2)
    r = model(input)
