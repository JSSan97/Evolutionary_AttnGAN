import torch
import torch.nn as nn
from miscc.config import cfg
from models.ca_net import CA_NET

# Used for text to image generation
class Generator(nn.Module):
    def __init__(self, ngf=128, output_nc=3, z_dim=100, img_size=64):
        super(Generator, self).__init__()

        self.z_dim = cfg.GAN.Z_DIM
        self.ngf = ngf
        self.output_nc = output_nc

        self.embed_dim = 256 ##  cfg.TEXT.EMBEDDING_DIM

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

    def forward(self, z_code, text_embedding, should_print=False):
        # Form batch size by 128 vector
        # projected_embed = self.projection(embed_vector)
        # if (should_print == True):
        #     print("embed_vector")
        #     print(embed_vector)
        #     print("projected_embed")
        #     print(projected_embed.shape)
        #     print(projected_embed)
        #

        c_code, mu, logvar = self.ca_net(text_embedding)

        ## squeezed_projected_embed = projected_embed.unsqueeze(2).unsqueeze(3)

        # Concatenate noise and text encoding
        latent_vector = torch.cat([c_code, z_code], 1)

        output = self.model(latent_vector.view(-1, self.latent_dim, 1, 1))

        return output, c_code, mu, logvar

# Use for text to image generation
class Discriminator(nn.Module):
    def __init__(self, ndf=128, input_nc=3, img_size=128):
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

        self.cnn_model = nn.Sequential(*seq)
        fc = [nn.Linear(4 * 4 * self.ndf, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, inp, c_code):
        x_intermediate = self.cnn_model(inp)
        # Concatenate intermediate value with embedding

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        x = torch.cat((c_code, x_intermediate), 1)

        x = x.view(-1, 4 * 4 * self.ndf)
        x = self.fc(x)
        return x