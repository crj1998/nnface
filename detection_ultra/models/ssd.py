
from typing import List, Tuple

import torch
import torch.nn as nn


from models.backbone import MobileNet
from models.module import SeperableConv2d

class SSD(nn.Module):
    """Compose a SSD model using the given components.
    """
    def __init__(self, 
        num_classes: int, 
        backbone: nn.ModuleList, 
        source_layer_indexes: List[int],
        extras: nn.ModuleList, 
        clf_heads: nn.ModuleList,
        reg_heads: nn.ModuleList, 
        device=None
    ):

        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = clf_heads
        self.regression_headers = reg_heads

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confs = []
        locs = []
        header_index = 0
        start_layer_index = 0
        end_layer_index = 0
        for end_layer_index in self.source_layer_indexes:
            for layer in self.backbone[start_layer_index: end_layer_index]:
                x = layer(x)
            conf, loc = self.compute_header(header_index, x)
            # print(header_index, x.shape, conf.shape, loc.shape)
            confs.append(conf)
            locs.append(loc)
            header_index += 1
            start_layer_index = end_layer_index

        for layer in self.backbone[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            conf, loc = self.compute_header(header_index, x)
            # print(header_index, x.shape, conf.shape, loc.shape)
            confs.append(conf)
            locs.append(loc)
            header_index += 1

        confs = torch.cat(confs, dim=1)
        locs = torch.cat(locs, dim=1)

        return confs, locs

    def compute_header(self, i, x):
        conf: torch.Tensor = self.classification_headers[i](x)
        conf = conf.permute(0, 2, 3, 1).contiguous()
        conf = conf.view(conf.size(0), -1, self.num_classes)

        loc: torch.Tensor = self.regression_headers[i](x)
        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(loc.size(0), -1, 4)

        return conf, loc


def build_model(num_classes, arch='slim', device=None):
    backbone = MobileNet(num_classes, arch=arch)
    backbone_model = backbone.model  # disable dropout layer

    source_layer_indexes = [8, 11, 13]
    extras = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(backbone.channels * 16, backbone.channels * 4, kernel_size=1),
            nn.ReLU(),
            SeperableConv2d(backbone.channels * 4, backbone.channels * 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    ])

    reg_heads = nn.ModuleList([
        SeperableConv2d(backbone.channels *  4, 3 * 4, kernel_size=3, padding=1),
        SeperableConv2d(backbone.channels *  8, 2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(backbone.channels * 16, 2 * 4, kernel_size=3, padding=1),
        nn.Conv2d(backbone.channels * 16, 3 * 4, kernel_size=3, padding=1)
    ])

    clf_heads = nn.ModuleList([
        SeperableConv2d(backbone.channels *  4, 3 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(backbone.channels *  8, 2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(backbone.channels * 16, 2 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(backbone.channels * 16, 3 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD(num_classes, backbone_model, source_layer_indexes, extras, clf_heads, reg_heads, device=device)


if __name__ == '__main__':
    model = build_model(2, 'rfb')
    x = torch.randn(4, 3, 256, 320)
    with torch.no_grad():
        y = model(x)
