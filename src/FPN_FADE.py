from collections import OrderedDict

import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.misc import Conv2dNormActivation

from FADE_H2L import FADE


class FADEFeaturePyramidNetwork(FeaturePyramidNetwork):
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__(in_channels_list, out_channels, extra_blocks)

        # Remove the existing top-down connections (original skip connections)
        del self.inner_blocks
        del self.layer_blocks

        # Re-define inner_blocks and layer_blocks to match the in_channels_list
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.fade_modules = nn.ModuleList()

        for idx, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")

            # Define 1x1 convolutions (inner blocks) to reduce the number of channels in the lateral feature maps
            # from their original depth (`in_channels`) to a fixed depth (`out_channels`).
            # This ensures all feature maps have a consistent depth (e.g., 256 channels) for further processing.
            inner_block_module = Conv2dNormActivation(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                padding=0,
                norm_layer=nn.BatchNorm2d, 
                activation_layer=None
            )
            self.inner_blocks.append(inner_block_module)

            # Define 3x3 convolutions (layer blocks) to refine the merged feature maps.
            # These feature maps result from combining the lateral feature map (from the backbone)
            # with the top-down feature map (from higher pyramid levels).
            # 
            # - The 3x3 kernel provides a larger receptive field than 1x1 convolutions,
            #   enabling better spatial refinement by incorporating neighboring pixel information.
            # - Padding of 1 ensures the spatial dimensions of the input are preserved.
            # - The depth of the output remains consistent (`out_channels`, e.g., 256),
            #   which is crucial for ensuring compatibility across all FPN levels.
            layer_block_module = Conv2dNormActivation(
                out_channels,
                out_channels,
                kernel_size=3,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None
            )
            self.layer_blocks.append(layer_block_module)

            # Add FADE modules except for the highest level
            # - FADE modules replace the additive skip connections in the original FPN.
            # - Each FADE module adaptively combines the lateral feature map (`en`) with the top-down feature map (`de`)
            #   using a gating mechanism and upsampling via CARAFE.
            # - The highest level does not require a FADE module since it does not have a top-down input.
            if idx < len(in_channels_list) - 1:
                fade_module = FADE(out_channels, out_channels)
                self.fade_modules.append(fade_module)


    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): Feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): Feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x_values = list(x.values())
        
        # Process the highest feature level
        # - The highest level does not have a top-down input, so only apply a 1x1 convolution (inner_blocks)
        #   followed by a 3x3 convolution (layer_blocks) for refinement.
        last_inner = self.inner_blocks[-1](x_values[-1])  # Reduce channels using 1x1 convolution
        results = []  # Initialize results list
        results.insert(0, self.layer_blocks[-1](last_inner))  # Refine the feature map with a 3x3 convolution

        # Process remaining feature levels in a top-down manner
        for idx in range(len(x_values) - 2, -1, -1):  # Iterate from second-highest to lowest feature level
            inner_lateral = self.inner_blocks[idx](x_values[idx])  # Reduce lateral feature map channels
            fade_module = self.fade_modules[idx]  # Get the corresponding FADE module for this level
            last_inner = fade_module(en=inner_lateral, de=last_inner)  # Combine lateral and top-down features adaptively
            results.insert(0, self.layer_blocks[idx](last_inner))  # Refine the combined feature map with a 3x3 convolution

        # Optionally process extra layers for features outside the backbone-defined levels
        if self.extra_blocks is not None:
            results = self.extra_blocks(results, x_values, names)

        # Create an OrderedDict to match the input feature names with their processed feature maps
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
