import torch
import torch.nn as nn
import torchvision.models._utils as _utils

from modules import FPN, SSH



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, backbone, in_channels, out_channels, return_layers):
        super(RetinaFace,self).__init__()

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_list = [
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]

        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        fpn_num = 3
        anchor_num = 2
        self.ClassHead = nn.ModuleList(
            [ClassHead(out_channels, anchor_num) for _ in range(fpn_num)]
        )
        self.BboxHead = nn.ModuleList(
            [BboxHead(out_channels, anchor_num) for _ in range(fpn_num)]
        )
        self.LandmarkHead = nn.ModuleList(
            [LandmarkHead(out_channels, anchor_num) for _ in range(fpn_num)]
        )

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        box_reg = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        obj_cls = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_reg = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        return box_reg, obj_cls, ldm_reg