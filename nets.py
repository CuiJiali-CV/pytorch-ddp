import torch.nn as nn
import torch

class reshape(nn.Module):
    def __init__(self, *args):
        super(reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class _netG(nn.Module):
    def __init__(self, z1_dim):
        super().__init__()
        ngf = 64
        self.input_z_dim = z1_dim
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(self.input_z_dim, ngf * 16, 8, 1, 0, bias=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(ngf * 4, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        assert z.size(1) == self.input_z_dim
        z = z.view(-1, self.input_z_dim, 1, 1)
        mu = self.decode(z) # 3, 32, 32
        return mu

class _netI(nn.Module):
    def __init__(self, z1_dim):
        super().__init__()
        nif = 64
        self.to_z_dim = z1_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, nif*4, 3, 1, 1), # 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(nif * 4, nif * 8, 4, 2, 1), # 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(nif * 8, nif * 16, 4, 2, 1), # 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(nif * 16, self.to_z_dim*2, 8, 1, 0),  # self.to_z_dim*2
        )

    def forward(self, x):
        logits = self.encode(x).squeeze() # (bs, z1 dim)
        mu, log_sig = logits.chunk(2, dim=1)
        sig = torch.exp(log_sig) + 1e-2
        return mu, sig