import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from utils.bbox import xywh2xyxy, xyxy2xywh

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, boxes):
        return TF.to_tensor(img), torch.from_numpy(boxes).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img, boxes):
        return TF.normalize(img, self.mean, self.std, self.inplace), boxes


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class ColorJitter(nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    """

    def __init__(self, p, brightness=0., contrast=0., saturation=0., hue=0.):
        super().__init__()
        self.p = p
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get the parameters for the randomized transform to be applied on image.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, img, boxes):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        if torch.rand(1) < self.p:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = TF.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = TF.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = TF.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = TF.adjust_hue(img, hue_factor)

        return img, boxes


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s

class RandomHorizontalFlip(nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, boxes):
        if torch.rand(1) < self.p:
            img = TF.hflip(img)
            boxes[:, 0] = 1.0 - boxes[:, 0]

        return img, boxes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomCrop(torch.nn.Module):
    """Crop the given image at a random location.
    size: (h, w)
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        assert isinstance(size, (int, tuple))
        self.size = (size, size) if isinstance(size, int) else size
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, boxes):
        _, H, W = TF.get_dimensions(img)

        xyxy_boxes = xywh2xyxy(boxes)
        xyxy_boxes[:, [0, 2]] *= W
        xyxy_boxes[:, [1, 3]] *= H

        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)
            left, top = self.padding[:2]
            xyxy_boxes[:, [0, 2]] += left
            xyxy_boxes[:, [1, 3]] += top
            _, H, W = TF.get_dimensions(img)

        # pad the width/height if needed
        if self.pad_if_needed and (W < self.size[1] or H < self.size[0]):
            left, top = max(0, self.size[1] - W), max(0, self.size[0] - H)
            padding = [left, top]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
            _, H, W = TF.get_dimensions(img)
            xyxy_boxes[:, [0, 2]] += left
            xyxy_boxes[:, [1, 3]] += top

        th, tw = self.size
        if th/H > tw/W:
            h = torch.randint(int(0.4*H), H, size=(1,)).item()
            w = int(tw / th * h)
        else:
            w = torch.randint(int(0.4*W), W, size=(1,)).item()
            h = int(th / tw * w)
        x = torch.randint(0, W-w, size=(1,)).item()
        y = torch.randint(0, H-h, size=(1,)).item()

        center = (xyxy_boxes[:, :2]+xyxy_boxes[:, 2:]) / 2 - torch.FloatTensor([[x, y]]).expand_as(xyxy_boxes[:, :2])
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = xyxy_boxes[mask.expand_as(boxes)].view(-1, 4)
        if boxes_in.size(0) > 0:
            boxes_in -= torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)
            boxes_in[:, [0, 2]] = boxes_in[:, [0, 2]].clamp(0, w) / w
            boxes_in[:, [1, 3]] = boxes_in[:, [1, 3]].clamp(0, h) / h
            boxes = xyxy2xywh(boxes_in)
            img = TF.crop(img, y, x, h, w)
        else:
            boxes = xyxy2xywh(xyxy_boxes)
            boxes[:, [0, 2]] /= W
            boxes[:, [1, 3]] /= H
        img = TF.resize(img, self.size)
        return img, boxes


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"
