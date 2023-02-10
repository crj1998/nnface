
import math
from itertools import product

import torch

class PriorBox(object):
    def __init__(self, min_sizes, strides, imgsz=640, clip=False):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.strides = strides
        self.clip = clip
        self.imgsz = imgsz
        self.feature_maps = [[math.ceil(self.imgsz[0]/s), math.ceil(self.imgsz[1]/s)] for s in self.strides]

    def __call__(self):
        anchors = []
        for min_sizes, s in zip(self.min_sizes, self.strides):
            fh, fw = math.ceil(self.imgsz[0] / s), math.ceil(self.imgsz[1] / s)
            for i, j in product(range(fh), range(fw)):
                for msz in min_sizes:
                    mw = msz / self.imgsz[1]
                    mh = msz / self.imgsz[0]
                    cx = (j + 0.5) * s / self.imgsz[1]
                    cy = (i + 0.5) * s / self.imgsz[0]
                    anchors.append([cx, cy, mw, mh])

        output = torch.Tensor(anchors)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.imgsz[1]
                    s_ky = min_size / self.imgsz[0]
                    dense_cx = [x * self.strides[k] / self.imgsz[1] for x in [j + 0.5]]
                    dense_cy = [y * self.strides[k] / self.imgsz[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
