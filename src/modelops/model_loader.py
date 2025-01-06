import torch
from torchvision import models
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import fasterrcnn_resnet50_fpn


import importlib.util
spec = importlib.util.spec_from_file_location(
    "FPN_FADE", "/content/drive/MyDrive/Colab Notebooks/cmpe593/term-project/src/FPN_FADE.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FPN_FADE = module.FADEFeaturePyramidNetwork


def create_baseline_model(num_classes=91, pretrained=True, model_path=None, device='cpu'):
    """
    Create a baseline Faster R-CNN model with ResNet-50 backbone.

    Args:
        num_classes (int): Number of classes including the background.
        pretrained (bool): Whether to use COCO-pretrained weights.
        model_path (str, optional): Path to the trained model checkpoint.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The baseline Faster R-CNN model.
    """
    if pretrained:
        print("Loading baseline model with COCO-pretrained weights...")
        # Fully pretrained Faster R-CNN model (COCO weights)
        model = fasterrcnn_resnet50_fpn(weights='COCO_V1', num_classes=num_classes)
    else:
        print("Loading baseline model which was produced from my own training...")
        # No pretrained weights (train from scratch)
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

        # Load custom-trained weights if model_path is provided
        if model_path:
            print(f"Loading baseline model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

    print("Baseline model loaded.")
    return model


def create_custom_FADE_model(device, model_path=None, num_classes=91):
    """Create a custom Faster R-CNN model with a FADE FPN."""
    print("Creating custom FADE model...")

    # Define a pretrained ResNet-50 backbone
    backbone = resnet50(pretrained=True)

    # Replace fully connected and average pooling layers with identity operations
    # to preserve spatial feature maps for object detection tasks.
    backbone.fc = torch.nn.Identity()
    backbone.avgpool = torch.nn.Identity()

    # Multi-scale feature extraction using intermediate layers
    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # Create the custom Feature Pyramid Network (FPN) with FADE
    # 
    # `in_channels_list`: The number of channels for the feature maps output 
    # by `layer2`, `layer3`, and `layer4` of the ResNet backbone. These values 
    # correspond to the architecture of ResNet-50:
    #   - `512` for `layer2` (conv3_x)
    #   - `1024` for `layer3` (conv4_x)
    #   - `2048` for `layer4` (conv5_x)
    # These layers provide multi-scale feature maps needed for object detection.
    # 
    # `out_channels`: The number of output channels for all levels of the FPN.
    #   - Standardized to `256` to balance computational efficiency and 
    #     representational capacity. This choice is widely adopted in FPN-based
    #     object detectors and originates from the original FPN design by Lin et al. (2017).
    in_channels_list = [512, 1024, 2048]
    out_channels = 256
    fpn_fade = FPN_FADE(in_channels_list, out_channels)

    # Combine the backbone with the custom FPN
    backbone_with_fpn = BackboneWithFPN(
        backbone=backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=None
    )
    backbone_with_fpn.fpn = fpn_fade

    # Define anchor generator
    anchor_sizes = ((32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Create the Faster R-CNN model with the custom backbone
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    if model_path:
        print(f"Loading custom FADE model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    print("Custom FADE model loaded.")
    return model


def load_model(model_type, model_path=None, device='cpu', num_classes=91, pretrained=True, eval_mode=True):
    """
    Load a Faster R-CNN model (baseline or custom FADE) for object detection.

    Args:
        model_type (str): 'baseline' or 'custom_FADE'.
        model_path (str, optional): Path to the state dictionary for the custom model.
        device (str): 'cpu' or 'cuda'.
        num_classes (int): Number of classes. Default is 91 (COCO classes).
        pretrained (bool): Whether to use pretrained weights. Default is True.
        eval_mode (bool): Whether to set the model to evaluation mode. Default is True.

    Returns:
        torch.nn.Module: The model ready for evaluation or further training.
    """
    if model_type == "baseline":
        model = create_baseline_model(num_classes=num_classes, pretrained=pretrained, model_path=model_path, device=device)
    elif model_type == "custom_FADE":
        model = create_custom_FADE_model(device, model_path, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if eval_mode:
        model.eval()
    model.to(device)
    print(f"Loaded {model_type} model for {'evaluation' if eval_mode else 'training'} on {device}.")
    return model


