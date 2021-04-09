# Modified from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomRotation.get_params

import math
import numbers
import random
import warnings
from enum import Enum
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int


class Rotation(torch.nn.Module):

    def __init__(
        self, angle, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.angle = angle

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        

        return F.rotate(img, self.angle, self.resample, self.expand, self.center, fill)
