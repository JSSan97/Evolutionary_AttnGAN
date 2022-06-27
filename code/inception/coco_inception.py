from inception import get_inception_score
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run inception scorer')
    parser.add_argument('--file_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/eval_coco_img_array.npy')
    parser.add_argument('--inception_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inception_attngan2.npy')
    args = parser.parse_args()
    return args


def save_imgs_as_numpy_array(path):
    eval = np.load(path, allow_pickle=True)
    eval = np.ndarray.tolist(eval)
    images = eval['validation_imgs']
    mean, std = get_inception_score(images=images)
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
    save_imgs_as_numpy_array(args.file_path)
