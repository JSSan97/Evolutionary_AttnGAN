import torch
import torch.nn as nn
from miscc.config import cfg
from models.ca_net import CA_NET

# Used for text to image generation
class Generator(nn.Module):
    def __init__(self, ngf=128, output_nc=3, img_size=64):
        super(Generator, self).__init__()

        self.z_dim = cfg.GAN.Z_DIM
        self.ngf = ngf
        self.output_nc = output_nc

        self.embed_dim = 100 ##  cfg.GAN.Z_DIM
        self.ca_net = CA_NET()

        self.latent_dim = self.z_dim + self.embed_dim

        if img_size == 32:
            seq = [nn.ConvTranspose2d(self.latent_dim, self.ngf * 4, 4, stride=1, padding=0, bias=True),
                   nn.BatchNorm2d(self.ngf * 4),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf * 2),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf, self.output_nc, 4, stride=2, padding=(1, 1)),
                   nn.Tanh()]

        if img_size == 64:
            seq = [nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, stride=1, padding=0, bias=True),
                   nn.BatchNorm2d(self.ngf * 8),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf * 4),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf * 2),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf, self.output_nc, 4, stride=2, padding=(1, 1)),
                   nn.Tanh()]

        if img_size == 128:
            seq = [nn.ConvTranspose2d(self.latent_dim, self.ngf * 16, 4, stride=1, padding=0, bias=True),
                   nn.BatchNorm2d(self.ngf * 16),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf * 8),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf * 4),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf * 2),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ngf),
                   nn.ReLU(),
                   nn.ConvTranspose2d(self.ngf, self.output_nc, 4, stride=2, padding=(1, 1)),
                   nn.Tanh()]

        self.model = nn.Sequential(*seq)

    def forward(self, z_code, text_embedding):
        c_code, mu, logvar = self.ca_net(text_embedding)
        latent_vector = torch.cat([c_code, z_code], 1)
        output = self.model(latent_vector.view(-1, self.latent_dim, 1, 1))

        return output, c_code, mu, logvar

# Use for text to image generation
class Discriminator(nn.Module):
    def __init__(self, ndf=128, input_nc=3, img_size=64):
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.input_nc = input_nc

        if img_size == 32:
            seq = [nn.Conv2d(self.input_nc, self.ndf, 4, stride=2, padding=(1, 1), bias=True),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf, self.ndf * 2, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 2),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 4),
                   nn.LeakyReLU(0.2)]

        if img_size == 64:
            seq = [nn.Conv2d(self.input_nc, self.ndf, 4, stride=2, padding=(1, 1), bias=True),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf, self.ndf * 2, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 2),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 4),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 8),
                   nn.LeakyReLU(0.2)]

        if img_size == 128:
            seq = [nn.Conv2d(self.input_nc, self.ndf, 4, stride=2, padding=(1, 1), bias=True),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf, self.ndf * 2, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 2),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 4),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 8),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, stride=2, padding=(1, 1), bias=True),
                   nn.BatchNorm2d(self.ndf * 16),
                   nn.LeakyReLU(0.2)
                   ]

        self.netD_1 = nn.Sequential(*seq)

        self.projected_embed_dim = 100
        self.netD_2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 16 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp, c_code):
        x_intermediate = self.netD_1(inp)
        replicated_embed = c_code.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        x = torch.cat([replicated_embed, x_intermediate], 1)
        x = self.netD_2(x)

        return x