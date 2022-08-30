from __future__ import print_function
from six.moves import range

import copy
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import weights_init, load_params, copy_G_params, count_parameters
from datasets import prepare_data
from miscc.losses import discriminator_loss, evo_generator_loss, KL_loss, discriminator_loss_with_logits, \
    get_word_and_sentence_loss
import time
import os

from algorithms.trainer import GenericTrainer


class ImprovedEvoTraining(GenericTrainer):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        super().__init__(output_dir, data_loader, n_words, ixtoword)

        # List of mutation counts per epoch
        self.minimax_list, self.least_squares_list, self.heuristic_list, self.crossover_list = self.load_mutation_count()
        print(self.minimax_list, self.least_squares_list, self.heuristic_list, self.crossover_list)

        self.MSE_loss = torch.nn.MSELoss()
        self.criterion_MAE = nn.L1Loss(reduction='none')
        self.crossover_size = 1

    def load_mutation_count(self):
        if cfg.EVO.RECORD_MUTATION and os.path.isfile(cfg.EVO.RECORD_MUTATION):
            mutations = np.load(cfg.EVO.RECORD_MUTATION, allow_pickle=True)
            mutations = np.ndarray.tolist(mutations)
            minimax = mutations['minimax']
            least_squares = mutations['least_squares']
            heuristic = mutations['heuristic']
            crossover = mutations['crossover']
            return minimax, least_squares, heuristic, crossover
        else:
            return [], [], [], []

    def save_mutation_count(self):
        if cfg.EVO.RECORD_MUTATION:
            mutations = {}
            mutations['minimax'] = self.minimax_list
            mutations['least_squares'] = self.least_squares_list
            mutations['heuristic'] = self.heuristic_list
            mutations['crossover'] = self.crossover_list
            np.save(cfg.EVO.RECORD_MUTATION, mutations)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()

        print(count_parameters(netG))
        print(count_parameters(netsD[0]))
        print(count_parameters(netsD[1]))
        print(count_parameters(netsD[2]))

        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
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
            mutation_dict = {
                'minimax': 0,
                'least_squares': 0,
                'heuristic': 0,
                'crossover': 0,
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

                ###########################################################
                # (2) Evolutionary Phase: Update G Networks and Select Best
                ###########################################################
                noise.data.normal_(0, 1)
                fake_imgs, selection, netG, optimizerG, G_logs, errG_total = self.evolution_phase(
                    netG, netsD, optimizerG, image_encoder,
                    real_labels, fake_labels,
                    words_embs, sent_emb, match_labels,
                    cap_lens, class_ids, mask, noise)

                mutation_dict[selection] = mutation_dict[selection] + 1
                # print(mutation_dict)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                eval_size = int(cfg.TRAIN.BATCH_SIZE // cfg.EVO.DISCRIMINATOR_UPDATES)
                self.set_requires_grad_value(netsD, True)

                for d in range(cfg.EVO.DISCRIMINATOR_UPDATES):
                    for i in range(len(netsD)):
                        eval_gen_imgs = fake_imgs[i][d * eval_size: (d+1)*eval_size]
                        eval_real_imgs = imgs[i][d * eval_size: (d+1)*eval_size]
                        eval_sent_emb = sent_emb[d * eval_size: (d+1)*eval_size]
                        eval_real_labels = real_labels[d * eval_size: (d+1)*eval_size]
                        eval_fake_labels = fake_labels[d * eval_size: (d+1)*eval_size]

                        netsD[i].zero_grad()
                        errD = discriminator_loss(netsD[i], eval_real_imgs, eval_gen_imgs,
                                                  eval_sent_emb, eval_real_labels, eval_fake_labels)
                        # backward and update parameters
                        errD.backward()
                        optimizersD[i].step()
                        errD_total += errD
                        D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update Params
                ######################################################
                step += 1
                gen_iterations += 1

                # update parameters
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                    print("Most recent mutation : {}".format(selection))
                    print(mutation_dict)

                # save images
                if gen_iterations % 300 == 0:
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

            self.minimax_list.append(mutation_dict['minimax'])
            self.least_squares_list.append(mutation_dict['least_squares'])
            self.heuristic_list.append(mutation_dict['heuristic'])
            self.crossover_list.append(mutation_dict['crossover'])

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)
                self.save_mutation_count()

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)
        self.save_mutation_count()

    def forward(self, noise, netG, sent_emb, words_embs, mask):
        fake_imgs, att_maps, mu, logvar = netG(noise, sent_emb, words_embs, mask)

        return fake_imgs, att_maps, mu, logvar

    def evolution_phase(self, netG, netsD, optimizerG, image_encoder,
                        real_labels, fake_labels,
                        words_embs, sent_emb, match_labels,
                        cap_lens, class_ids, mask, current_noise):

        # 3 types of mutations
        if cfg.EVO.MUTATIONS == 1:
            mutations = ['heuristic']
        else:
            mutations = ['minimax', 'heuristic', 'least_squares']

        fitness = []
        mutate_pop = []
        mutate_optim = []
        mutate_critics = []
        gen_imgs_list = []

        g_logs_list = []
        errG_list = []

        # Parent of evolution cycle
        G_candidate_dict = copy.deepcopy(netG.state_dict())
        optG_candidate_dict = copy.deepcopy(optimizerG.state_dict())

        noise_mutate = current_noise.data.normal_(0, 1)
        noise_crossover = noise_mutate.normal_(0, 1)

        # Go through every mutation
        for m in range(cfg.EVO.MUTATIONS):
            # Perform Variation
            netG.load_state_dict(G_candidate_dict)
            optimizerG.load_state_dict(optG_candidate_dict)
            optimizerG.zero_grad()

            fake_imgs, _, mu, logvar = self.forward(current_noise, netG, sent_emb, words_embs, mask)

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
                mutation_gen_images, _, _, _ = self.forward(noise_mutate, netG, sent_emb, words_embs, mask)
                w_loss, s_loss = get_word_and_sentence_loss(image_encoder, mutation_gen_images[-1], words_embs,
                                                            sent_emb, match_labels, cap_lens, class_ids,
                                                            real_labels.size(0))
            f, gen_critic = self.fitness_score(netsD[-1], mutation_gen_images[-1], sent_emb, w_loss.item(),
                                               s_loss.item())

            mutate_pop.append(copy.deepcopy(netG.state_dict()))
            mutate_optim.append(copy.deepcopy(optimizerG.state_dict()))
            mutate_critics.append(gen_critic)
            gen_imgs_list.append(mutation_gen_images)
            fitness.append(f)
            errG_list.append(errG_total)
            g_logs_list.append(G_logs)

        ########## CROSSOVER ############
        # Initialize crossover population
        crossover_pop = []
        crossover_optim = []

        sorted_groups = self.sort_groups_by_fitness(range(cfg.EVO.MUTATIONS), fitness)

        for i in range(self.crossover_size):
            first, second, _ = sorted_groups[i % len(sorted_groups)]
            netG, optimizerG, c_loss = self.distilation_crossover(netG, optimizerG, netsD, noise_mutate, mutate_pop[first],
                                                          mutate_optim[first], mutate_critics[first],
                                                          gen_imgs_list[first][-1], mutate_critics[second],
                                                          gen_imgs_list[second][-1],
                                                          crossover_pop, crossover_optim, sent_emb, words_embs, mask)

        g_logs_list[first] += 'crossover_loss: %.2f ' % c_loss.item()
        G_logs = g_logs_list[first]
        errG_total = errG_list[first]

        for i in range(self.crossover_size):
            netG.load_state_dict(crossover_pop[i])
            with torch.no_grad():
                crossover_gen_images, _, _, _ = self.forward(noise_crossover, netG, sent_emb, words_embs, mask)
                w_loss, s_loss = get_word_and_sentence_loss(image_encoder, crossover_gen_images[-1], words_embs,
                                                            sent_emb, match_labels, cap_lens, class_ids,
                                                            real_labels.size(0))
            crossover_f, _ = self.fitness_score(netsD[-1], crossover_gen_images[-1], sent_emb, w_loss, s_loss)
            fitness.append(crossover_f)
            gen_imgs_list.append(crossover_gen_images)

        ########## SELECTION ############
        top_n = np.argsort(fitness)[-1:]
        index = top_n[0]

        if index >= len(mutations):
            eval_imgs = gen_imgs_list[index]
            index = index - len(mutations)
            gene = copy.deepcopy(crossover_pop[index])
            gene_optimizer = copy.deepcopy(crossover_optim[index])
            selected = 'crossover'
        else:
            eval_imgs = gen_imgs_list[index]
            gene = copy.deepcopy(mutate_pop[index])
            gene_optimizer = copy.deepcopy(mutate_optim[index])
            selected = mutations[index % len(mutations)]

        netG.load_state_dict(gene)
        optimizerG.load_state_dict(gene_optimizer)

        return eval_imgs, selected, netG, optimizerG, G_logs, errG_total

    def distilation_crossover(self, netG, optimizerG, netsD, noise, gene1, gene1_optim, gene1_critic, gene1_sample,
                              gene2_critic, gene2_sample,
                              offspring, offspring_optim, sent_emb, words_embs, mask):
        self.set_requires_grad_value(netsD, False)

        netG.load_state_dict(gene1)
        optimizerG.load_state_dict(gene1_optim)

        optimizerG.zero_grad()

        eps = 0.0

        # Take the best of samples in gene1 and best of samples in gene2 and put together a list of images
        fake_batch = torch.cat((gene1_sample[gene1_critic - gene2_critic > eps],
                                gene2_sample[gene2_critic - gene1_critic >= eps])).detach()
        # Also take the best input noise
        noise_batch = torch.cat((noise[gene1_critic - gene2_critic > eps], noise[gene2_critic - gene1_critic >= eps]))

        # Ensure sent, word and mask are the same
        new_sent_emb = torch.cat((sent_emb[gene1_critic - gene2_critic > eps],
                                  sent_emb[gene2_critic - gene1_critic >= eps])).detach()

        new_word_emb = torch.cat((words_embs[gene1_critic - gene2_critic > eps],
                                  words_embs[gene2_critic - gene1_critic >= eps])).detach()

        new_mask = torch.cat((mask[gene1_critic - gene2_critic > eps],
                              mask[gene2_critic - gene1_critic >= eps])).detach()

        offspring_batch, _, _, _ = self.forward(noise_batch, netG, new_sent_emb, new_word_emb, new_mask)
        offspring_batch = offspring_batch[-1]

        # Offspring Update
        policy_loss = self.MSE_loss(offspring_batch, fake_batch)
        policy_loss.backward()
        optimizerG.step()

        offspring.append(copy.deepcopy(netG.state_dict()))
        offspring_optim.append(copy.deepcopy(optimizerG.state_dict()))

        return netG, optimizerG, policy_loss

    def sort_groups_by_fitness(self, genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i + 1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    def fitness_score(self, netD, fake_imgs, sent_emb, w_loss, s_loss):
        # Get fitness scores of the last stage, i.e. assess 256x256
        fake_features = netD(fake_imgs)
        cond_output = netD.COND_DNET(fake_features, sent_emb, logits=False)
        uncond_output = netD.UNCOND_DNET(fake_features, logits=False)


        Fc = (cfg.EVO.QUALITY_CONDITIONAL_LAMBDA * cond_output)
        Fu = (cfg.EVO.QUALITY_UNCONDITIONAL_LAMBDA * uncond_output)

        Fq = Fc + Fu

        Fd = torch.empty(0)
        if cfg.CUDA:
            Fd = Fd.cuda()

        # Diversity fitness score
        comp_size = 5  # 1/3/5/30
        for i in range(comp_size):
            shuffle_ids = torch.randperm(fake_imgs.size(0))
            disorder_samples = fake_imgs[shuffle_ids]
            loss = self.criterion_MAE(fake_imgs, disorder_samples)
            # loss = self.criterion_MSE(gen_samples, disorder_samples).sqrt_()
            loss_samples = loss.reshape(fake_imgs.size(0), -1).mean(1).unsqueeze(0)
            Fd = torch.cat((Fd, loss_samples))
        Fd = Fd.mean(0)

        Fw = -cfg.EVO.WORD_LOSS_LAMBDA * w_loss
        Fs = -cfg.EVO.SENTENCE_LOSS_LAMBDA * s_loss

        F_critic = Fq + (cfg.EVO.DIVERSITY_LAMBDA * Fd) + Fw + Fs
        f = (Fq + (cfg.EVO.DIVERSITY_LAMBDA * Fd) + Fw + Fs).mean().item()

        # print("F: {}, Fu: {}, Fc: {} Fq: {}, Fd: {}, Fw: {}, Fs: {}".format(f,
        #                                                      Fu.mean().item(),
        #                                                      Fc.mean().item(),
        #                                                      Fq.mean().item(),
        #                                                      Fd.mean().item(), Fw, Fs))

        return f, F_critic
