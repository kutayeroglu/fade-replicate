"""## Integrate modified FPN into Faster R-CNN by modifying the backbone"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models._utils import IntermediateLayerGetter


# Load a pre-trained ResNet-50 model and extract its layers up to the last convolutional block
backbone = resnet50(pretrained=True)

# Remove the fully connected layer and average pool
backbone.fc = nn.Identity()
backbone.avgpool = nn.Identity()

# Define the return layers to extract features for the FPN
return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}

from torchvision.models._utils import IntermediateLayerGetter

# Create the backbone with the specified return layers
backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

# Define the number of input channels for each feature map that the FPN will receive
in_channels_list = [512, 1024, 2048]  # Corresponding to layer2, layer3, layer4
out_channels = 256  # The number of channels in the FPN output feature maps

# Create the custom FADE FPN
fpn = FADEFeaturePyramidNetwork(
    in_channels_list=in_channels_list,
    out_channels=out_channels,
    extra_blocks=None
)

# Combine the backbone and the custom FPN
backbone_with_fpn = BackboneWithFPN(
    backbone=backbone,
    return_layers=return_layers,
    in_channels_list=in_channels_list,
    out_channels=out_channels,
    extra_blocks=None
)

# Replace the default FPN with our custom FADE FPN
backbone_with_fpn.fpn = fpn

# Create the Faster R-CNN model with the custom backbone
num_classes = 91  # Adjust as needed

from torchvision.models.detection.rpn import AnchorGenerator

# Define anchor sizes for each feature map
anchor_sizes = ((32,), (64,), (128,))  # One size per feature map level

# Define aspect ratios for each feature map
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

# Create the custom anchor generator
anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=aspect_ratios
)

model = FasterRCNN(
    backbone=backbone_with_fpn,
    num_classes=91,
    rpn_anchor_generator=anchor_generator
)

# Test the model with a sample input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
sample_image = torch.randn(3, 800, 800)
sample_image = sample_image.to(device)

with torch.no_grad():
    outputs = model([sample_image])

# Print the outputs
print(outputs)