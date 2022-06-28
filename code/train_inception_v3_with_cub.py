import argparse
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torchvision import transforms

from datasets import CubInceptionDataset
from miscc.config import cfg, cfg_from_file
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='Train Inception Model on Cubs')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='code/cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100, help='Training Epochs')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch Training')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='Learning Rate')
    parser.add_argument('--results_path', type=str, default='/content/drive/MyDrive/Github/Evolutionary_AttnGAN/models')

    args = parser.parse_args()
    return args

def train_loop(dataloader, model, criterion, optimizer, device):
    size = len(dataloader.dataset)
    loss_ep = 0
    correct = 0

    data_iter = iter(dataloader)
    step = 0
    while step < len(dataloader):
        data = data_iter.next()
        imgs, class_ids = data

        imgs = imgs.to(device=device)
        targets = class_ids.to(device=device)

        print(imgs.shape)
        print(len(targets))

        predictions, aux_outputs = model(imgs)
        print(predictions.shape)
        print(aux_outputs.shape)

        loss1 = criterion(predictions, targets)
        print(loss1.shape)
        print(loss1.item())

        loss2 = criterion(aux_outputs, targets)
        print(loss2.shape)
        print(loss2.item())

        loss = loss1 + 0.4 * loss2
        print(loss.shape)
        print(loss.item())

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ep += loss.item()

        # Add correct
        correct += (predictions.argmax(1) == targets).type(torch.float).sum().item()
        step += 1

    avg_loss = loss_ep / len(dataloader)
    accuracy = (correct / size) * 100

    print("Train Accuracy: {:>0.1f}%, Avg Loss: {:.3f}".format(accuracy, avg_loss))
    return avg_loss,

def test_loop(dataloader, model, criterion, device, logger):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            predictions = model(data)
            test_loss += criterion(predictions, targets).item()
            correct += (predictions.argmax(1) == targets).type(torch.float).sum().item()

    avg_loss = (test_loss / num_batches)
    accuracy = (correct / size) * 100
    logger.info("Test Accuracy: {:>0.1f}%, Avg Loss: {:.3f}".format(accuracy, avg_loss))
    return avg_loss, accuracy


def main(args):
    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    split_dir, bshuffle = 'train', True

    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    ## Load Model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False, num_classes=200)
    model.fc = nn.Linear(2048, 200)
    model.AuxLogits = InceptionAux(768, 200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    ## Load Dataset
    image_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = CubInceptionDataset(cfg.DATA_DIR, split=split_dir, transform=image_transform)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    ## Training Configs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    ## Train Epochs
    start_time = time.time()
    for epoch in range(args.epochs):
        print("---- Epoch {} ----".format(epoch + 1))
        train_avg_loss, train_accuracy = train_loop(dataloader, model, criterion, optimizer, device)

        # test_avg_loss, test_accuracy = test_loop(test_loader, model, criterion, device, logger)

        current_time = time.time() - start_time
        print("Current Training Time: {}".format(current_time))

        if epoch % 20:
            model_filename = "inceptinv3_{}.pth".format(epoch + 1)
            torch.save(model.state_dict(), "{}/{}".format(args.results_path, model_filename))

    end_time = time.time()
    total_train_time = end_time - start_time
    print("Total Training Time: {}".format(total_train_time))

    # Save Model
    model_filename = "inceptionV3_{}.pth".format(epoch + 1)
    torch.save(model.state_dict(), "{}/{}".format(args.results_path, model_filename))


if __name__ == "__main__":
    ## Convert all images to 256x256
    args = parse_args()
    main(args)


