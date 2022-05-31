import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import time
import numpy as np

from datasets import prepare_data
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, load_params, copy_G_params, load_image_from_tensor
from miscc.losses import KL_loss
from models.dcgan import Generator, Discriminator
from models.attngan_model import RNN_ENCODER
from PIL import Image
from torch.autograd import Variable

class DCGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words):
        self.criterion = nn.BCELoss()
        self.n_words = n_words
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def build_models(self):
        ## Load NetG
        netD = Discriminator()
        netD.apply(weights_init)
        ## Load NetG
        netG = Generator()
        netG.apply(weights_init)

        epoch = 0

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1

            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                s_tmp = Gname[:Gname.rfind('/')]
                Dname = '%s/netD.pth' % (s_tmp)
                print('Load D from: ', Dname)
                state_dict = \
                    torch.load(Dname, map_location=lambda storage, loc: storage)
                netD.load_state_dict(state_dict)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
            text_encoder = text_encoder.cuda()
        return [text_encoder, netG, netD, epoch]

    def define_optimizers(self, netG, netD):
        optimizerD = optim.Adam(netD.parameters(),
                                lr=cfg.TRAIN.DISCRIMINATOR_LR,
                                betas=(0.5, 0.999))

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizerD

    def discriminator_loss(self, netD, real_imgs, fake_imgs, c_code, real_labels, fake_labels):
        real_out = netD(inp=real_imgs, c_code=c_code).view(-1)
        fake_out = netD(inp=fake_imgs, c_code=c_code).view(-1)

        errD_real = self.criterion(real_out, real_labels)
        errD_fake = self.criterion(fake_out, fake_labels)
        errD = errD_real + errD_fake
        return errD

    def generator_loss(self, netD, fake_imgs, c_code, real_labels):
        output = netD(inp=fake_imgs.detach(), c_code=c_code).view(-1)
        errG = self.criterion(output, real_labels)

        logs = 'g_loss: %.2f ' % (errG.item())
        return errG, logs

    def save_model(self, netG, avg_param_G, netD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)


        torch.save(netD.state_dict(),
            '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def save_img_results(self, netG, noise, sent_emb,
                         epoch, name='current'):
        # Save images
        fake_imgs, _, _, _ = netG(z_code=noise, text_embedding=sent_emb)
        fullpath = '%s/G_%s_%d_%d.png'\
            % (self.image_dir, name, epoch)
        load_image_from_tensor(torchvision.utils.make_grid(fake_imgs.cpu()), show=False, save=True, output=fullpath)

    def set_requires_grad_value(self, model, brequires):
        for p in model.parameters():
            p.requires_grad = brequires


    def train(self):
        text_encoder, netG, netD, start_epoch = self.build_models()

        avg_param_G = copy_G_params(netG)
        optimizerG, optimizerD = self.define_optimizers(netG, netD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netD, True)
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                _, sent_emb = text_encoder(captions, cap_lens, hidden)
                sent_emb = sent_emb.detach()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, c_code, mu, logvar = netG(z_code=noise, text_embedding=sent_emb)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                netD.zero_grad()
                real_imgs = imgs[1] # torch.Size([64, 3, 128, 128])
                errD = self.discriminator_loss(netD, real_imgs, fake_imgs,
                                          c_code, real_labels, fake_labels)
                # backward and update parameters
                errD.backward()
                optimizerD.step()
                errD_total += errD
                D_logs += 'errD: %.2f ' % (errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netD, False)
                fake_imgs, c_code, mu, logvar = netG(z_code=noise, text_embedding=sent_emb)
                netG.zero_grad()
                errG, G_logs = self.generator_loss(netD, fake_imgs, c_code, real_labels)
                kl_loss = KL_loss(mu, logvar)
                errG += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG.backward()
                optimizerG.step()

                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb, epoch, name='average')
                    load_params(netG, backup_para)

            end_t = time.time()

            print('''[%d/%d][%d]
                     Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netD, epoch)

        self.save_model(netG, avg_param_G, netD, self.max_epoch)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            # Build and load the generator
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            netG = Generator()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()

            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM

                with torch.no_grad():
                    captions = Variable(torch.from_numpy(captions))
                    cap_lens = Variable(torch.from_numpy(cap_lens))

                    captions = captions.cuda()
                    cap_lens = cap_lens.cuda()

                for i in range(1):  # 16
                    with torch.no_grad():
                        noise = Variable(torch.FloatTensor(batch_size, nz))
                        noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    _, sent_emb = text_encoder(captions, cap_lens, hidden)

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(z_code=noise, text_encoder=sent_emb)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)
