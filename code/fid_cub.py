import argparse
import torch
import torch.nn as nn
import os
import random
import torch.utils.data as data
import numpy as np

from torch.autograd import Variable
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from PIL import Image
from scipy.linalg import sqrtm
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from cub_inception_score import write_sub_filenames, CubEvalDataset
from train_inception_v3_with_cub import InceptionAux
from torchvision import transforms

TEST_ONLY_CLASSES = ['001.', '004.', '006.', '008.', '009.', '014.', '023.', '029.', '031.', '033.', '034.', '035.', '036.', '037.', '038.', '043.', '049.', '051.', '053.', '066.', '072.', '079.', '083.', '084.', '086.', '091.', '095.', '096.', '098.', '101.', '102.', '103.', '112.', '114.', '119.', '121.', '130.', '135.', '138.', '147.', '156.', '163.', '165.', '166.', '180.', '183.', '185.', '186.', '187.', '197.']

class CubDataset(data.Dataset):
    def __init__(self, ground_truth_dir, filenames, eval_class=''):
    # def __init__(self, ground_truth_dir, eval_dir, filenames, eval_class=''):
        self.ground_truth_dir = ground_truth_dir
        # self.eval_dir = eval_dir
        self.filenames = filenames
        self.image_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ])

        self.eval_class = eval_class
        self.all_keys = self.get_all_keys()
        self.length = len(self.all_keys)

    def get_all_keys(self):
        with open(self.filenames, 'r') as f:
            all_keys = f.readlines()
            all_keys = [x.replace('\n', '') for x in all_keys]

            if self.eval_class:
                # Remove extension and sentence num e.g. _s0.png
                all_keys = [x[:-7] for x in all_keys if self.eval_class in x]
            else:
                all_keys = [x[:-7] for x in all_keys]

            # Remove duplicates
            all_keys = list(dict.fromkeys(all_keys))

        return all_keys

    # def get_eval_img(self, key):
    #     sentence_num = random.randint(0, 9)
    #     eval_img = os.path.join(self.eval_dir, "{}_s{}.png".format(key, sentence_num))
    #     return eval_img

    def __getitem__(self, index):
        key = self.all_keys[index].replace('\n', '')
        ground_truth_img = os.path.join(self.ground_truth_dir, "{}.jpg".format(key))
        ground_truth_img = self.image_transform(Image.open(ground_truth_img).convert('RGB'))

        # eval_img = self.get_eval_img(key)
        # while not os.path.exists(eval_img):
        #     print("This image does not exists: {}".format(eval_img))
        #     eval_img = self.get_eval_img(key)
        #
        # eval_img = self.image_transform(Image.open(eval_img).convert('RGB'))
        #
        # return ground_truth_img, eval_img
        return ground_truth_img

    def __len__(self):
        return self.length


def parse_args():
    parser = argparse.ArgumentParser(description='Run inception scorer')
    parser.add_argument('eval_imgs_dir', type=str)
    parser.add_argument('--ground_truth_dir', type=str, default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/data/birds/CUB_200_2011/images')
    parser.add_argument('--inception_v3_model', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inceptionV3_Cl50_100.pth')
    parser.add_argument('--classes', type=int,
                        default=50)
    parser.add_argument('--results_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/fid_scores/birds_attngan2_600.npy')
    args = parser.parse_args()
    return args

def load_model(classes, model_path):
    ## Load Model and modify classes
    model = torch.hub.load('pytorch/vision:v0.11.0', 'inception_v3', pretrained=False, num_classes=classes)
    model.fc = nn.Linear(2048, classes)
    model.AuxLogits = InceptionAux(768, classes)

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Extract train and evaluation layers from the mdoel
    train_nodes, eval_nodes = get_graph_node_names(model)

    # Remove last layer
    return_nodes = eval_nodes[:-1]

    # create a feature extractor for each intermediary layer
    feat_inception = create_feature_extractor(model, return_nodes=return_nodes)
    feat_inception.to(device)

    return feat_inception

def fid(args, model, class_name):
    if not os.path.isfile('eval_filenames.txt'):
        write_sub_filenames(args.eval_imgs_dir)

    # gt_dataset = CubDataset(args.ground_truth_dir, args.eval_imgs_dir, 'eval_filenames.txt', eval_class=class_name)
    gt_dataset = CubDataset(args.ground_truth_dir, 'eval_filenames.txt', eval_class=class_name)
    eval_dataset = CubEvalDataset(args.eval_imgs_dir, 'eval_filenames.txt', eval_class=class_name)

    shuffle = False

    gt_data_loader = torch.utils.data.DataLoader(
        gt_dataset, batch_size=len(gt_dataset),
        drop_last=True, shuffle=shuffle)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=len(gt_dataset),
        drop_last=True, shuffle=shuffle)

    activation1, activation2 = get_feature_vector(model, gt_data_loader, eval_data_loader)

    print(activation2.shape)
    fid_score = calculate_fid(activation1, activation2)

    return fid_score

def get_feature_vector(model, gt_data_loader, eval_data_loader):
    gt_data_iter = iter(gt_data_loader)
    # ground_truths, eval_imgs = data_iter.next()
    ground_truths = gt_data_iter.next()
    ground_truths = ground_truths.cuda()
    ground_truths = Variable(ground_truths).cuda()

    output_feat_1 = model(ground_truths)
    vec_feat_1 = output_feat_1['flatten'].cpu().detach().numpy()

    # eval_data_iter = iter(eval_data_loader)
    vec_feats_2 = []
    # i = 0
    for i, sample_batch in enumerate(eval_data_loader, 0):
        # eval_imgs = eval_data_iter.next()
        eval_imgs = sample_batch
        eval_imgs = eval_imgs.cuda()
        eval_imgs = Variable(eval_imgs).cuda()
        output_feat_2 = model(eval_imgs)
        vec_feat_2 = output_feat_2['flatten'].cpu().detach().numpy()
        vec_feats_2.append(vec_feat_2)
        i += 1

    return vec_feat_1, np.concatenate(vec_feats_2)

def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.classes, args.inception_v3_model)
    class_fid_scores = {}
    for class_id in TEST_ONLY_CLASSES:
        print("===== Class: {} =====".format(class_id))
        fid_score = fid(args, model, class_id)
        print(fid_score)
        class_fid_scores[class_id] = fid
        np.save(args.results_path, class_fid_scores)
