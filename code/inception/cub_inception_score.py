import numpy as np
import argparse
import os
import torch.nn as nn
import torch
import math
from torch.autograd import Variable

from torch.nn import functional as F
from code.train_inception_v3_with_cub import InceptionAux
from scipy.stats import entropy

def parse_args():
    parser = argparse.ArgumentParser(description='Run inception scorer')
    parser.add_argument('--file_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/eval_coco_img_array.npy')
    parser.add_argument('--inception_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inception_attngan2.npy')
    parser.add_argument('--inception_v3_model', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inceptionV3_100.pth')
    parser.add_argument('--batch_size', type=int,
                        default=24)
    parser.add_argument('--splits', type=int,
                        default=1)
    args = parser.parse_args()
    return args


def inception(args):
    eval = np.load(args.path, allow_pickle=True)
    eval = np.ndarray.tolist(eval)
    images = eval['validation_imgs']

    mean, std = get_inception_score(images=images, model=args.inception_v3_model, splits=args.splits)
    print("==== Mean ====")
    print(mean)
    print("==== Standard Deviation ====")
    print(std)

    mean_list, std_list = load_inception_scores(args.inception_path)
    mean_list.append(mean)
    std_list.append(std)

    inception = {}
    inception['mean'] = mean_list
    inception['std'] = std_list
    np.save(args.inception_path, inception)

def get_inception_score(images, model_path, batch_size, splits):
    ## Load Model and modify classes
    model = torch.hub.load('pytorch/vision:v0.11.0', 'inception_v3', pretrained=False, num_classes=200)
    model.fc = nn.Linear(2048, 200)
    model.AuxLogits = InceptionAux(768, 200)

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dtype = torch.cuda.FloatTensor
    # Get predictions
    preds = np.zeros((len(images), 200))
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        x = up(x)
        x = model(x)
        return F.softmax(x).data.cpu().numpy()

    num_batches = math.ceil(len(images) / batch_size)
    i = 0

    while i < num_batches:
        eval_imgs = images[i * batch_size: (i+1) * batch_size]
        eval_imgs = eval_imgs.type(dtype)
        eval_imgsv = Variable(eval_imgs)
        batch_size_i = eval_imgs.size()[0]
        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(eval_imgsv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (len(images) // splits): (k+1) * (len(images) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def load_inception_scores(inception_path):
    if os.path.isfile(inception_path):
        inception = np.load(inception_path, allow_pickle=True)
        inception = np.ndarray.tolist(inception)
        mean_list = inception['mean']
        std_list = inception['std']
        return mean_list, std_list
    else:
        return [], []

if __name__ == "__main__":
    args = parse_args()
    inception(args)
