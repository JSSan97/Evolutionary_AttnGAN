# Evo-AttnGAN and IE-AttnGAN

Evo-AttnGAN introduces the evolutionary training from Evolutionary-GAN (E-GAN) and Improved Evolutionary GAN
(IE-GAN) to the text-to-image based GAN: AttnGAN. New fitness components are introduced in the fitness function 
and changes are made to the loss functions.

### Dependencies
python 3.6+

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`


**Dataset and Metadata**

1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) and save them to `data/` (Credit goes to Original Authors)
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`


**Training (Optionally Skip and use DAMSM and Pretrained Models Below)**
- Pre-train DAMSM models:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0` 

- Train models for the bird dataset:
  - `python main.py --cfg cfg/bird_attn2.yml --gpu 0`
  - `python main.py --cfg cfg/bird_evo_attn2.yml --gpu 0`
  - `python main.py --cfg cfg/bird_ie_attn2.yml --gpu 0`

- `*.yml` files are example configuration files for training/evaluation the models.


**Pretrained Models from the original authors**
- [DAMSM for bird](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V). Download and save it to `DAMSMencoders/`. 
- [AttnGAN for bird](https://drive.google.com/open?id=1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig). Download and save it to `models/`

**My pretrained models**
- All generator models can be found here. Download and save it to `models/`.  https://drive.google.com/drive/folders/1AwvcwkXaQZ2yPcAQubUlvtBG_T0auS1i?usp=sharing
- CUB Inception Model (For inception scoring and FID). Download and save it to `models/`: https://drive.google.com/file/d/1ZkTuUaaQb3CVh3XpQi6_w1PYkv3CtV4r/view?usp=sharing
  - Alternatively train your own CUB inception model. See 'train_inception_v3_with_cub.py'

**Evaluating Models**


First run models to output images. The output images will be in a directory of where the generator model is (ensure to configure NET_G parameter of cfg file to point to the generator model).
  - `python main.py --cfg cfg/bird_eval_experiments/birds_attngan2/eval_bird_attn_700.yml --gpu 0`
  - `python main.py --cfg cfg/bird_eval_experiments/birds_evo_attngan2/eval_bird_attn_700.yml --gpu 0`
  - `python main.py --cfg cfg/bird_eval_experiments/birds_ie_attngan2/eval_bird_attn_700.yml --gpu 0`
  - `python main.py --cfg cfg/bird_eval_experiments/birds_ie_attngan2_tuned_05/eval_bird_attn_700.yml --gpu 0`
  - `python main.py --cfg cfg/bird_eval_experiments/birds_ie_attngan2_tuned_05/eval_bird_attn_700.yml --gpu 0`

See .yml file to see configurations. Note that B_VALIDATION=True runs captions from all test classes. if B_VALIDATION=True, EVAL_EVERY_CAPTION runs all captions of the test classes, otherwise it samples a caption for every image. If B_VALIDATION=False, then the program uses the captions in example_captions.txt to be used as input.
Ensure NET_G parameter points to the path of the generator model

Inception scoring. Change arguments as necessary (see cub_inception_score.py), the first argument is the path to the output images. Note that pred_path saves predictions into an npy file, toggle --use_pred to use an existing npy file. This is to save time as
calculating the inception score can take long, especially if evaluating images of all captions in the test classes.
- `!python3 code/cub_inception_score.py /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/birds_attngan2/birds_attngan2_700/valid/single --pred_path /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/inception_predictions/birds_attngan2_700.npy --splits=10 --batch_size=20 --inception_v3_model /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inceptionV3_Cl50_100.pth`

Inception scoring of each test class. E.g.
- `!python3 code/cub_inception_score.py /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/birds_ie_attngan2/birds_ie_attngan2_700/valid/single --eval_single_class=True --splits=10 --batch_size=20 --inception_v3_model /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/inceptionV3_Cl50_100.pth --inception_path /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/inception_scores/classes_ie_attngan2_700.npy`

FID scoring of each test class. E.g.
- `!python3 code/fid_cub.py /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/birds_ie_attngan2/birds_ie_attngan2_700/valid/single --results_path /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/fid_scores/birds_ie_attngan2_700.npy --batch_size=30`

R-Precision. First command sets up the data for R-Precision calculation in ./RP_DATA. Second commands evaluates to retrieve R-Precision score. E.g.
- `!python3 code/build_RPData.py /content/drive/MyDrive/Github/Evolutionary_AttnGAN/models/birds_experiments/birds_ie_attngan2/birds_ie_attngan2_700/valid/single -t /content/drive/MyDrive/Github/Evolutionary_AttnGAN/data/birds/text`
- `!python3 code/eval_RP.py ./RP_DATA  /content/drive/MyDrive/Github/Evolutionary_AttnGAN/data/birds/attngan_captions.pickle --cfg /content/drive/MyDrive/Github/Evolutionary_AttnGAN/code/cfg/bird_eval_experiments/birds_ie_attngan2/eval_bird_attn_700.yml`


### AttnGAN
See original Pytorch implementation: https://github.com/taoxugit/AttnGAN
and paper [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He.
