import torch
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN


import importlib.util
spec = importlib.util.spec_from_file_location(
    "FPN_FADE", "/content/drive/MyDrive/Colab Notebooks/cmpe593/term-project/src/FPN_FADE.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FPN_FADE = module.FADEFeaturePyramidNetwork


def create_baseline_model(num_classes=91):
    """Create a standard Faster R-CNN model with a ResNet-50 backbone."""
    return FasterRCNN(
        backbone=resnet50(pretrained=True),
        num_classes=num_classes
    )


def create_custom_FADE_model(device, model_path, num_classes=91):
    """Create a custom Faster R-CNN model with a FADE FPN."""

    # Define the custom backbone
    backbone = resnet50(pretrained=True)
    backbone.fc = torch.nn.Identity()
    backbone.avgpool = torch.nn.Identity()

    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # Create the custom FPN
    in_channels_list = [512, 1024, 2048]
    out_channels = 256
    fpn = FPN_FADE(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=None
    )

    # Combine the backbone with the custom FPN
    backbone_with_fpn = BackboneWithFPN(
        backbone=backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=None
    )
    backbone_with_fpn.fpn = fpn

    # Define anchor generator
    anchor_sizes = ((32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # Create the Faster R-CNN model with the custom backbone
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    # Load the custom state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def load_model(model_type, model_path=None, device='cpu', num_classes=91):
    """
    Load a Faster R-CNN model (baseline or custom FADE) for object detection.

    Args:
        model_type (str): Type of the model to load. Options: 'baseline', 'custom_FADE'.
        model_path (str, optional): Path to the state dictionary for the custom model.
        device (str): Device to load the model on ('cpu' or 'cuda').
        num_classes (int): Number of classes for the model. Default is 91.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.

    Raises:
        ValueError: If an unsupported model type is provided or required arguments are missing.
    """
    if model_type == "baseline":
        model = create_baseline_model(num_classes)
    elif model_type == "custom_FADE":
        if not model_path:
            raise ValueError("A valid model_path must be provided for 'custom_FADE'.")
        model = create_custom_FADE_model(device, model_path, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()
    model.to(device)
    print(f"Loaded {model_type} model on {device}.")
    return model
