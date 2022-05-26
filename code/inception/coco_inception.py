from inception import get_inception_score

def save_imgs_as_numpy_array(path):
    eval = np.load(path, allow_pickle=True)
    eval = np.ndarray.tolist(eval)
    images = eval['validation_imgs']
    mean, std = get_inception_score(images=images)
    print(mean)
    print(std)

if __name__ == "__main__":
    save_imgs_as_numpy_array('/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/eval_coco_img_array.npy')
