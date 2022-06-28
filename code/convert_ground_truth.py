import argparse
import os
import torchvision.transforms as transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Preliminary, convert all images to 256x256 for FID')
    parser.add_argument('--image_path', type=str, default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/data/birds/CUB_200_2011/images/')
    parser.add_argument('--output_path', type=str, default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/data/birds/CUB_200_2011/images_resized/')
    args = parser.parse_args()
    return args

def main(image_path, output_path):
    transform = transforms.Resize(size=(256, 256))

    # Look through all image folders
    for image_folder in os.listdir(image_path):
        # Get full path of image folder
        directory = os.path.join(image_path, image_folder)

        # Make new directory for image in output path
        output_directory = os.path.join(output_path, image_folder)
        if not os.path.isfile(output_directory):
            os.mkdir(output_directory)

        for image_file in os.listdir(directory):
            img_path = os.path.join(directory, image_file)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)

            image_to_png = image_file.replace(".jpg", ".png")
            output_img_path = os.path.join(output_directory, image_to_png)
            print("Saving : {}".format(output_img_path))

            img.save(output_img_path)


if __name__ == "__main__":
    ## Convert all images to 256x256
    args = parse_args()
    main(args.image_path, args.output_path)


