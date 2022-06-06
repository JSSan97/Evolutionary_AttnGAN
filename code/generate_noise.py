import torch
import argparse
import numpy as np
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Fixed Random Noise, Used in all Tests')
    parser.add_argument('--file_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/fixed_noise.npy')
    parser.add_argument('--batch_size', type=int,
                        default=5)

    args = parser.parse_args()
    return args


def generate_noise(bs, nz, path):
    fixed_noise = Variable(torch.FloatTensor(bs, nz).normal_(0, 1))
    noise = {}
    noise['fixed_noise'] = fixed_noise
    np.save(path, noise)

if __name__ == "__main__":
    args = parse_args()
    generate_noise(args.batch_size, args.nz, args.path)
