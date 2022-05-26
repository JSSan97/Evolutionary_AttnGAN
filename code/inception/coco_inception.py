from inception import get_inception_score
from miscc.config import cfg

def save_imgs_as_numpy_array():
    mean, std = get_inception_score(images=cfg.B_VALIDATION_IMG_ARRAY)
    print(mean)
    print(std)
