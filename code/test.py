import torch.nn as nn
import torch

if __name__ == "__main__":
    real_tensor = torch.ones(64, 1)
    criteria = nn.BCELoss()
    loss_real = criteria(real_tensor, real_tensor)
    print(real_tensor)
    print(loss_real)