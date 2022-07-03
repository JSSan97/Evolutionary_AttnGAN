import numpy as np
import argparse
import os
import torch.nn as nn
import torch
import math

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from train_inception_v3_with_cub import InceptionAux
from scipy.stats import entropy
import torch.utils.data as data

class CubEvalDataset(data.Dataset):
    def __init__(self, data_dir, filenames):
        self.data_dir = data_dir
        self.filenames = filenames
        self.image_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ])

    def get_image_path(self, index):
        with open(self.filenames, 'r') as f:
            all_texts = f.readlines()
            filename = all_texts[index].replace('\n', '')
        image_path = os.path.join(self.data_dir, filename)
        return image_path

    def __getitem__(self, index):
        image_path = self.get_image_path(index)
        image = self.image_transform(Image.open(image_path).convert('RGB'))
        return image

    def __len__(self):
        return 29280


def parse_args():
    parser = argparse.ArgumentParser(description='Run inception scorer')
    parser.add_argument('eval_imgs_dir', type=str)
    # parser.add_argument('--file_path', type=str,
    #                     default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/eval_coco_img_array.npy')
    parser.add_argument('--inception_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inception_attngan2.npy')
    parser.add_argument('--inception_v3_model', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inceptionV3_Cl50_100.pth')
    parser.add_argument('--classes', type=int,
                        default=50)
    parser.add_argument('--batch_size', type=int,
                        default=20)
    parser.add_argument('--splits', type=int,
                        default=10)
    args = parser.parse_args()
    return args

def write_sub_filenames(eval_imgs_dir):
    '''
    Images are in seperate class folders, so we want to map the image to filenames
    Used in dataloadeer.
    '''
    # Look through all class image folders
    eval_filenames = []
    for image_folder in os.listdir(eval_imgs_dir):
        # Get full path of class image folder
        class_directory = os.path.join(eval_imgs_dir, image_folder)
        for image_file in os.listdir(class_directory):
            key = os.path.join(image_folder, image_file) # 001.bird_black/my_bird1.png
            eval_filenames.append(key)

        with open("eval_filenames.txt", mode='w') as f:
            f.write('\n'.join(eval_filenames))


def inception(args):
    if not os.path.isfile('eval_filenames.txt'):
        print("Writing class_folder/filename to text file")
        write_sub_filenames(args.eval_imgs_dir)

    # eval = np.load(args.file_path, allow_pickle=True)
    # eval = np.ndarray.tolist(eval)
    # images = eval['validation_imgs']
    dataset = CubEvalDataset(args.eval_imgs_dir, 'eval_filenames.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=True, shuffle=True)

    mean, std = get_inception_score(data_loader=dataloader, model_path=args.inception_v3_model, batch_size=args.batch_size, splits=args.splits, classes=args.classes, num_images=len(dataset))
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

def get_inception_score(data_loader, model_path, batch_size, splits, classes, num_images):
    ## Load Model and modify classes
    model = torch.hub.load('pytorch/vision:v0.11.0', 'inception_v3', pretrained=False, num_classes=classes)
    model.fc = nn.Linear(2048, classes)
    model.AuxLogits = InceptionAux(768, classes)

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print("Num of images {}".format(num_images))
    preds = np.zeros((len(data_loader), args.classes))

    # Get predictions

    def get_pred(x):
        # x = up(x)
        pred = model(x)
        return torch.nn.functional.softmax(pred[0], dim=0).data.cpu().numpy()
        # return F.softmax(pred).data.cpu().numpy()

    data_iter = iter(data_loader)
    i = 0
    while i < len(data_loader):
        eval_imgs = data_iter.next()
        eval_imgs = eval_imgs.cuda()
        eval_imgs = Variable(eval_imgs).cuda()
        preds[i * batch_size:(i+1) * batch_size] = get_pred(eval_imgs)
        i += 1

    # dtype = torch.cuda.FloatTensor
    # num_batches = math.ceil(len(images) / batch_size)
    # i = 0
    #
    # while i < num_batches:
    #     eval_imgs = images[i * batch_size: (i+1) * batch_size]
    #     # print(eval_imgs.shape)
    #     eval_imgs = torch.stack(eval_imgs)
    #
    #     eval_imgs = eval_imgs.type(dtype)
    #     eval_imgsv = Variable(eval_imgs)
    #     batch_size_i = eval_imgs.shape[0]
    #     preds[i * batch_size:(i * batch_size) + batch_size_i] = get_pred(eval_imgsv)
    #     i += 1

    print("Finished passing through model")
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (num_images // splits): (k+1) * (num_images // splits), :]
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
