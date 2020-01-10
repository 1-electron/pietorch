import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# fetch pretrained resnet
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
num_classes = 10  # cifar 10
model_ft.fc = nn.Linear(num_ftrs, num_classes)