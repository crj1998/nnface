import os
import cv2
import numpy as np
import torch


class Model(object):
    def __init__(self):
        self.model = None
    
    def _pre_process(self, f):
        assert os.path.exists(f) and os.path.isfile(f)
        # read image as np.ndarray
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        # convert BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # move the channnel dim to first
        im = im.transpose(2, 0, 1)
        # normalize
        im = im.astype(np.float32) / 255.
        # to torch.Tensor and become batch.
        im = torch.from_numpy(im).unsqueeze(dim=0)

        assert im.ndim == 4 and im.size(1) == 3

        return im

    def _post_process(self, outputs):
        return None

    @torch.no_grad()
    def inference(self, f):

        inputs = self._pre_process(f)
        outputs = self.model(inputs)
        outputs = self._post_process(outputs)

        return outputs

