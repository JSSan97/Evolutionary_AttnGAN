from __future__ import print_function
from six.moves import range

import copy
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import weights_init, load_params, copy_G_params
from datasets import prepare_data
from miscc.losses import discriminator_loss, evo_generator_loss, KL_loss, discriminator_loss_with_logits
import time
import os

from algorithms.trainer import GenericTrainer


class EvoTraining(GenericTrainer):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        super().__init__(output_dir, data_loader, n_words, ixtoword)

        # List of mutation counts per epoch
        self.minimax_list, self.least_squares_list, self.heuristic_list = self.load_mutation_count()

    def load_mutation_count(self):
        if cfg.EVO.RECORD_MUTATION and os.path.isfile(cfg.EVO.RECORD_MUTATION):
            mutations = np.load(cfg.EVO.RECORD_MUTATION, allow_pickle=True)
            mutations = np.ndarray.tolist(mutations)
            minimax = mutations['minimax']
            least_squares = mutations['least_squares']
            heuristic = mutations['heuristic']
            return minimax, least_squares, heuristic
        else:
            return [], [], []

    def save_mutation_count(self, mutation_dict):
        if cfg.EVO.RECORD_MUTATION:
            self.minimax_list.append(mutation_dict['minimax'])
            self.least_squares_list.append(mutation_dict['least_squares'])
            self.heuristic_list.append(mutation_dict['heuristic'])

            mutations = {}
            mutations['minimax'] = self.minimax_list
            mutations['least_squares'] = self.least_squares_list
            mutations['heuristic'] = self.heuristic_list
            np.save(cfg.EVO.RECORD_MUTATION, mutations)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()
        fake_imgs = None

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            mutation_dict = {
                'minimax': 0,
                'least_squares': 0,
                'heuristic': 0,
            }

            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                # On the first go, otherwise use images generated from best candidate from evolution phase
                if fake_imgs is None:
                    fake_imgs, _, _, _ = self.forward(noise, netG, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for d in range(cfg.EVO.DISCRIMINATOR_UPDATES):
                    for i in range(len(netsD)):
                        netsD[i].zero_grad()
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  sent_emb, real_labels, fake_labels)
                        # backward and update parameters
                        errD.backward()
                        optimizersD[i].step()
                        errD_total += errD
                        D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                step += 1
                gen_iterations += 1

                fake_imgs, mutation, netG, optimizerG, G_logs, errG_total = self.evolution_phase(
                    netG, netsD, optimizerG, image_encoder,
                    real_labels, fake_labels,
                    words_embs, sent_emb, match_labels,
                    cap_lens, class_ids, mask, noise, imgs)

                mutation_dict[mutation] = mutation_dict[mutation] + 1

                # update parameters
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                    print("Most recent mutation : {}".format(mutation))
                    print(mutation_dict)

                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            print('''Mutations in epoch [%d/%d][%d]'''
                  % (epoch, self.max_epoch, self.num_batches))
            print(mutation_dict)

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)
                self.save_mutation_count(mutation_dict)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)
        self.save_mutation_count(mutation_dict)

    def forward(self, noise, netG, sent_emb, words_embs, mask):
        fake_imgs, att_maps, mu, logvar = netG(noise, sent_emb, words_embs, mask)

        return fake_imgs, att_maps, mu, logvar

    def evolution_phase(self, netG, netsD, optimizerG, image_encoder,
                        real_labels, fake_labels,
                        words_embs, sent_emb, match_labels,
                        cap_lens, class_ids, mask, noise, real_imgs):

        # 3 types of mutations
        if cfg.EVO.MUTATIONS == 1:
            mutations = ['heuristic']
        else:
            mutations = ['minimax', 'heuristic', 'least_squares']

        F_list = np.zeros(1)
        Fit_list = []
        G_list = []
        optG_list = []
        evalimg_list = []
        selected_mutation = []
        g_logs_list = []
        errG_list = []

        # Parent of evolution cycle
        G_candidate_dict = copy.deepcopy(netG.state_dict())
        optG_candidate_dict = copy.deepcopy(optimizerG.state_dict())

        count = 0

        # Go through every mutation
        for m in range(cfg.EVO.MUTATIONS):
            # Perform Variation
            netG.load_state_dict(G_candidate_dict)
            optimizerG.load_state_dict(optG_candidate_dict)
            optimizerG.zero_grad()

            fake_imgs, _, mu, logvar = self.forward(noise, netG, sent_emb, words_embs, mask)

            self.set_requires_grad_value(netsD, False)
            errG_total, G_logs = evo_generator_loss(netsD, image_encoder, fake_imgs,
                                                    real_labels, fake_labels,
                                                    words_embs, sent_emb, match_labels,
                                                    cap_lens, class_ids, mutations[m])

            kl_loss = KL_loss(mu, logvar)
            errG_total += kl_loss
            G_logs += 'kl_loss: %.2f ' % kl_loss.item()

            errG_total.backward()
            optimizerG.step()

            # Perform Evaluation
            with torch.no_grad():
                eval_fake_imgs, _, _, _ = self.forward(noise, netG, sent_emb, words_embs, mask)
            Fq, Fd = self.fitness_score(netsD, eval_fake_imgs, real_imgs, fake_labels, real_labels, sent_emb)
            F = Fq + cfg.EVO.DIVERSITY_LAMBDA * Fd

            # Perform selection
            if count < 1:
                F_list[count] = F
                Fit_list.append([Fq, Fd, F])
                G_list.append(copy.deepcopy(netG.state_dict()))
                optG_list.append(copy.deepcopy(optimizerG.state_dict()))
                evalimg_list.append(eval_fake_imgs)
                selected_mutation.append(mutations[m])
                g_logs_list.append(G_logs)
                errG_list.append(errG_total)
            else:
                fit_com = F - F_list
                if max(fit_com) > 0:
                    ids_replace = np.where(fit_com == max(fit_com))[0][0]
                    F_list[ids_replace] = F
                    Fit_list[ids_replace] = [Fq, Fd, F]
                    G_list[ids_replace] = copy.deepcopy(netG.state_dict())
                    optG_list[ids_replace] = copy.deepcopy(optimizerG.state_dict())
                    evalimg_list[ids_replace] = eval_fake_imgs
                    selected_mutation[ids_replace] = mutations[m]
                    g_logs_list[ids_replace] = G_logs
                    errG_list[ids_replace] = errG_total
            count += 1

        netG.load_state_dict(G_list[0])
        optimizerG.load_state_dict(optG_list[0])
        mutation_chosen = selected_mutation[0]
        eval_imgs = evalimg_list[0]
        logs = g_logs_list[0]
        errG_total = errG_list[0]

        return eval_imgs, mutation_chosen, netG, optimizerG, logs, errG_total

    def fitness_score(self, netsD, fake_imgs, real_imgs, fake_labels, real_labels, sent_emb):
        self.set_requires_grad_value(netsD, True)

        # Get fitness scores of the last stage, i.e. assess 256x256
        i = len(netsD) - 1

        eval_D, cond_eval_fake, uncond_eval_fake = \
            discriminator_loss_with_logits(netsD[i], real_imgs[i], fake_imgs[i], sent_emb,
                                           real_labels, fake_labels)

        # Quality fitness score
        # The unconditional evaluation determines whether the image is real or fake
        uncond_eval_fake = uncond_eval_fake.data.mean().cpu().numpy()
        # The conditional evaluation determines whether the image and the sentence match or not
        cond_eval_fake = cond_eval_fake.data.mean().cpu().numpy()
        # Quality fitness score
        Fq = (cfg.EVO.QUALITY_UNCONDITIONAL_LAMBDA * uncond_eval_fake) + \
             (cfg.EVO.QUALITY_CONDITIONAL_LAMBDA * cond_eval_fake)

        grad_outputs = torch.ones(eval_D.size()).cuda()

        if cfg.CUDA:
            grad_outputs = grad_outputs.cuda()

        # Diversity fitness score
        gradients = torch.autograd.grad(outputs=eval_D, inputs=netsD[i].parameters(),
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()

        return Fq, Fd
