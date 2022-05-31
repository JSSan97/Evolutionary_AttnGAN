from inception import get_inception_score
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run inception scorer')
    parser.add_argument('--file_path', type=str,
                        default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/eval_coco_img_array.npy')
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

if __name__ == "__main__":
    args = parse_args()
    save_imgs_as_numpy_array(args.file_path)
