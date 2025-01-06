from collections import OrderedDict

import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork

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
                continue

            # Define 1x1 convolutions (inner blocks) to reduce the number of channels in the lateral feature maps
            # from their original depth (`in_channels`) to a fixed depth (`out_channels`).
            # This ensures all feature maps have a consistent depth (e.g., 256 channels) for further processing.
            inner_block_module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.inner_blocks.append(inner_block_module)

            layer_block_module = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.layer_blocks.append(layer_block_module)

            # Add FADE modules except for the highest level
            if idx < len(in_channels_list) - 1:
                fade_module = FADE(out_channels, out_channels)
                self.fade_modules.append(fade_module)


    def forward(self, x):
      """
      Args:
          x (OrderedDict[Tensor]): Feature maps for each feature level.
      Returns:
          results (OrderedDict[Tensor]): Feature maps after FPN layers.
      """
      names = list(x.keys())
      x_values = list(x.values())
      last_inner = self.inner_blocks[-1](x_values[-1])
      results = []
      results.insert(0, self.layer_blocks[-1](last_inner))

      for idx in range(len(x_values) - 2, -1, -1):
          inner_lateral = self.inner_blocks[idx](x_values[idx])
          fade_module = self.fade_modules[idx]
          last_inner = fade_module(en=inner_lateral, de=last_inner)
          results.insert(0, self.layer_blocks[idx](last_inner))

      # Optionally add extra layers
      if self.extra_blocks is not None:
          results = self.extra_blocks(results, x_values, names)
      out = OrderedDict([(k, v) for k, v in zip(names, results)])
      return out
