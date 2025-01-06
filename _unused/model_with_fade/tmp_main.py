# Import libraries

## General purpose
"""

import os
import time
import logging

from tqdm import tqdm

"""## Torch"""

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator

"""# Mount Drive"""

from google.colab import drive

drive.mount('/content/drive')


















# # Define your model architecture
# model = FasterRCNN(backbone=backbone_with_fpn, num_classes=num_classes)
# model.load_state_dict(torch.load('model_epoch_X.pth'))
# model.to(device)
# model.eval()