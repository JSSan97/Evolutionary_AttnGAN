import numpy as np
import argparse
import os
import torch.nn as nn
import torch

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from train_inception_v3_with_cub import InceptionAux
import torch.utils.data as data

TEST_ONLY_CLASSES = ['001.', '004.', '006.', '008.', '009.', '014.', '023.', '029.', '031.', '033.', '034.', '035.', '036.', '037.', '038.', '043.', '049.', '051.', '053.', '066.', '072.', '079.', '083.', '084.', '086.', '091.', '095.', '096.', '098.', '101.', '102.', '103.', '112.', '114.', '119.', '121.', '130.', '135.', '138.', '147.', '156.', '163.', '165.', '166.', '180.', '183.', '185.', '186.', '187.', '197.']
TRAIN_ONLY_CLASSES = ['002.', '003.', '005.', '007', '010.', '011.', '012.', '013.', '015.', '016.', '017.', '018.', '019.', '020.', '021.', '022.', '024.', '025.', '026.', '027.', '028.', '030.', '032.', '039.', '040.', '041.', '042.', '044.', '045.', '046.', '047.', '048.', '050.', '052.', '054.', '055.', '056.', '057.', '058.', '059.', '060.', '061.', '062.', '063.', '064.', '065.', '067.', '068.', '069.', '070.',
                      '071.', '073.', '074.', '075.', '076.', '077.', '078.', '080.', '081.', '082.', '085.', '087.', '088.', '089.', '090.', '092.', '093.', '094.', '097.', '099.', '100.', '104.', '105.', '106.', '107.', '108.', '109.', '110.', '111.', '113.', '115.', '116.', '117.', '118.', '120.', '122.', '123.', '124.', '125.', '126.', '127.', '128.', '129.', '131.', '132.', '133.', '134.', '136.', '137.', '139.', '140.', '141.', '142.', '143.', '144.', '145.', '146.', '148.', '149.',
                      '150.', '151.', '152.', '153.', '154.', '155.', '157.', '158.', '159.', '160.', '161.', '162.', '164.', '167.', '168.', '169.', '170.', '171.', '172.', '173.', '174.', '175.', '176.', '177.', '178.', '179.', '181.', '182.', '184.', '188.', '189.', '190.', '191.', '192.', '193.', '194.', '195.', '196.', '198.', '199', '200']

class CubEvalDataset(data.Dataset):
    def __init__(self, data_dir, filenames, eval_class=''):
        self.data_dir = data_dir
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
            all_texts = f.readlines()
            if self.eval_class:
                all_texts = [x for x in all_texts if self.eval_class in x]

        return all_texts

    def get_image_path(self, index):
        filename = self.all_keys[index].replace('\n', '')
        image_path = os.path.join(self.data_dir, filename)
        return image_path

    def __getitem__(self, index):
        image_path = self.get_image_path(index)
        image = self.image_transform(Image.open(image_path).convert('RGB'))
        return image

    def __len__(self):
        return self.length


def parse_args():
    parser = argparse.ArgumentParser(description='Run inception scorer')
    parser.add_argument('eval_imgs_dir', type=str)
    parser.add_argument('--pred_path', type=str, default='')
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
    parser.add_argument('--use_pred', type=bool, default=False)
    parser.add_argument('--eval_single_class', type=bool, default=False)
    args = parser.parse_args()
    return args

def write_sub_filenames(eval_imgs_dir, filename, classes):
    '''
    Images are in seperate class folders, so we want to map the image to filenames
    Used in dataloadeer.
    '''
    # Look through all class image folders
    eval_filenames = []
    for image_folder in os.listdir(eval_imgs_dir):
        if [id for id in classes if id in image_folder]:
            # Get full path of class image folder
            class_directory = os.path.join(eval_imgs_dir, image_folder)
            print("Looking at directory {}".format(class_directory))
            for image_file in os.listdir(class_directory):
                key = os.path.join(image_folder, image_file) # 001.bird_black/my_bird1.png
                eval_filenames.append(key)

        with open(filename, mode='w') as f:
            f.write('\n'.join(eval_filenames))


def inception(args, model, eval_filenames, save=False, class_name=''):
    # eval = np.load(args.file_path, allow_pickle=True)
    # eval = np.ndarray.tolist(eval)
    # images = eval['validation_imgs']
    shuffle = False
    if class_name:
        shuffle = True
    dataset = CubEvalDataset(args.eval_imgs_dir, eval_filenames, eval_class=class_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=True, shuffle=shuffle)

    preds = get_predictions(data_loader=dataloader, model=model,
                                    batch_size=args.batch_size,
                                    classes=args.classes, num_images=len(dataset),
                                    use_pred=args.use_pred, pred_path=args.pred_path, save=save)

    print(len(dataset))
    print(preds.shape)
    mean, std = get_inception_score(preds, args.splits, len(dataset))
    print("==== Mean ====")
    print(mean)
    print("==== Standard Deviation ====")
    print(std)

    if save:
        mean_list, std_list = load_inception_scores(args.inception_path)
        mean_list.append(mean)
        std_list.append(std)

        inception = {}
        inception['mean'] = mean_list
        inception['std'] = std_list
        np.save(args.inception_path, inception)

    return mean, std

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
    return model


def get_predictions(data_loader, model, batch_size, classes, num_images, use_pred, pred_path, save):
    num_images = num_images - (num_images % batch_size)
    preds = np.zeros((num_images, classes))

    if not use_pred:
        print("Num of images {}".format(num_images))
        print("Num of batches {}".format(len(data_loader)))
        # Get predictions

        def get_pred(x):
            # x = up(x)
            with torch.no_grad():
                pred = model(x)
            return torch.nn.functional.softmax(pred[0], dim=0).data.cpu().numpy()
            # return F.softmax(pred).data.cpu().numpy()

        data_iter = iter(data_loader)
        i = 0
        #while i < 1:
        while i < len(data_loader):
            eval_imgs = data_iter.next()
            eval_imgs = eval_imgs.cuda()
            eval_imgs = Variable(eval_imgs).cuda()
            preds[i * batch_size:(i+1) * batch_size] = get_pred(eval_imgs)
            i += 1
            if(i % 100 == 0):
                print("Batches complete {}".format(i))

        if save and pred_path:
            print("Saving predictions")
            with open(pred_path, 'wb') as f:
                np.save(f, preds)

            print("Finished passing through model")
    else:
        print("Loading predictions from {}".format(pred_path))
        if pred_path:
            with open(pred_path, 'rb') as f:
                preds = np.load(pred_path)

    return preds

def get_inception_score(preds, splits, num_images):
    print("Computing IS")
    # Now compute the mean kl-div
    split_scores = []
    #
    # for k in range(splits):
    #     part = preds[k * (num_images // splits): (k+1) * (num_images // splits), :]
    #     py = np.mean(part, axis=0)
    #     scores = []
    #     for i in range(part.shape[0]):
    #         pyx = part[i, :]
    #         scores.append(entropy(pyx, py))
    #     split_scores.append(np.exp(np.mean(scores)))

    for k in range(splits):
        part = preds[k * (num_images // splits): (k+1) * (num_images // splits), :]
        kl = (part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        split_scores.append(np.exp(kl))

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

    model = load_model(args.classes, args.inception_v3_model)
    if args.classes == 50:
        eval_classes = TEST_ONLY_CLASSES
        filenames = 'eval_test_filenames.txt'

    elif args.classes == 150:
        eval_classes = TRAIN_ONLY_CLASSES
        filenames = 'eval_train_filenames.txt'

    if not os.path.isfile(filenames):
        print("Writing filenames in {}".format(filenames))
        write_sub_filenames(args.eval_imgs_dir, filenames, eval_classes)

    if args.eval_single_class:
        index = 0
        class_is_scores = {}

        for class_ids in eval_classes:
            print("Evaluating class {}".format(class_ids))
            mean, std = inception(args, model, filenames, save=False, class_name=class_ids)
            class_is_scores[class_ids] = (mean, std)
            index += 1
            np.save(args.inception_path, inception)


    else:
        _, _ = inception(args, model, filenames, save=True)
