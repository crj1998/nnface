import torch
import torch.nn as nn
import torch.nn.functional as F


from models.module import conv_bn, conv_dw, BasicRFB

# Receptive Field Block Net for Accurate and Fast Object Detection

class MobileNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, arch='slim'):
        super(MobileNet, self).__init__()
        channels = self.channels = 8 * 2
        if arch == 'slim':
            self.model = nn.Sequential(
                conv_bn(in_channels, channels, 2),  # 160*120
                conv_dw(channels     , channels *  2, 1),
                conv_dw(channels *  2, channels *  2, 2),  # 80*60
                conv_dw(channels *  2, channels *  2, 1),
                conv_dw(channels *  2, channels *  4, 2),  # 40*30
                conv_dw(channels *  4, channels *  4, 1),
                conv_dw(channels *  4, channels *  4, 1),
                conv_dw(channels *  4, channels *  4, 1),
                conv_dw(channels *  4, channels *  8, 2),  # 20*15
                conv_dw(channels *  8, channels *  8, 1),
                conv_dw(channels *  8, channels *  8, 1),
                conv_dw(channels *  8, channels * 16, 2),  # 10*8
                conv_dw(channels * 16, channels * 16, 1)
            )
        else:
            self.model = nn.Sequential(
                conv_bn(3, channels, 2),  # 160*120
                conv_dw(channels, channels * 2, 1),
                conv_dw(channels * 2, channels * 2, 2),  # 80*60
                conv_dw(channels * 2, channels * 2, 1),
                conv_dw(channels * 2, channels * 4, 2),  # 40*30
                conv_dw(channels * 4, channels * 4, 1),
                conv_dw(channels * 4, channels * 4, 1),
                BasicRFB(channels * 4, channels * 4, stride=1, scale=1.0),
                conv_dw(channels * 4, channels * 8, 2),  # 20*15
                conv_dw(channels * 8, channels * 8, 1),
                conv_dw(channels * 8, channels * 8, 1),
                conv_dw(channels * 8, channels * 16, 2),  # 10*8
                conv_dw(channels * 16, channels * 16, 1)
            )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



