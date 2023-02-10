import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

import torch
from torch.utils.data import Dataset


class WIDER(Dataset):
    cache_images = False
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        with open(txt_path,'r') as f:
            lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

        
        if self.cache_images:
            n = len(self.imgs_path)
            self.ims = [None] * n
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            fcn = lambda i: cv2.imread(self.imgs_path[i])
            results = ThreadPool(8).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.ims[i] = x
                b += self.ims[i].nbytes
                pbar.desc = f'Caching images ({b / gb:.1f}GB)'
            pbar.close()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image = self.ims[index] if self.cache_images else cv2.imread(self.imgs_path[index])

        target = self.words[index]
        annotations = np.zeros((0, 15))
        if len(target) == 0:
            return annotations
        for idx, label in enumerate(target):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y

            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        if self.preproc is not None:
            image, target = self.preproc(image, target)

        image, target = torch.from_numpy(image).float(), torch.from_numpy(target).float()
        return image, target

    @staticmethod
    def collate_fn(batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """

        imgs, tars = list(zip(*batch))

        # Resize images to input shape
        imgs = torch.stack(imgs, dim=0)

        return imgs, tars
