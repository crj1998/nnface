import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from data.augment import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize
except:
    from augment import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

def lbl2img_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}labels{os.sep}', f'{os.sep}images{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.jpg' for x in img_paths]

class WIDER(Dataset):
    cache_images = not (__name__ == '__main__')
    class_names = ('background', 'face')
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        f = glob.glob(f'{root}/labels/**/*.*', recursive=True)
        self.img_files = lbl2img_paths(f)  # labels
        self.lbl_files = f

        if self.cache_images:
            n = len(self.img_files)
            self.imgs = [None] * n
            self.lbls = [None] * n
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            fcn = lambda idx: (cv2.imread(self.img_files[idx]), np.loadtxt(self.lbl_files[idx]))
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, ncols=80)
            for i, (img, lbl) in pbar:
                self.imgs[i] = img
                self.lbls[i] = lbl
                b += self.imgs[i].nbytes
                pbar.desc = f'Caching images ({b / gb:.1f}GB)'
            pbar.close()

    def __len__(self):
        return len(self.lbl_files)

    def __getitem__(self, index):
        if self.cache_images:
            img, lbl = self.imgs[index], self.lbls[index]
        else:
            img, lbl = cv2.imread(self.img_files[index]), np.loadtxt(self.lbl_files[index])
        if lbl.ndim == 1:
            lbl = np.expand_dims(lbl, axis=0)
        lbl = lbl[:, 1:]
        if self.transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    @staticmethod
    def collate_fn(batch):
        # Drop invalid images
        # batch = [data for data in batch if data[1].shape[0]>0 ]

        imgs, lbls = list(zip(*batch))

        # Resize images to input shape
        imgs = torch.stack(imgs)

        # Add sample index to targets
        # for i, boxes in enumerate(lbls):
        #     boxes[:, 0] = i
        # lbls = torch.cat(lbls, dim=0)

        return imgs, lbls

def build_loader(root, split, batch_size, num_workers):
    cfg = {
        'batch_size': batch_size, 
        'num_workers': num_workers, 
        'collate_fn': WIDER.collate_fn
    }
    if split == 'train':
        transform = Compose([
            ToTensor(),
            RandomCrop((240, 320), padding=(120, 160), fill=0.5, pad_if_needed=False),
            RandomHorizontalFlip(p=0.5),
            # ColorJitter(0.75, 0.2, 0.2, 0.2, 0.2),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cfg.update({
            'shuffle': True, 
            'drop_last': True,
        })
    else:
        transform = Compose([
            ToTensor(),
            RandomCrop((240, 320), padding=0, fill=0.5, pad_if_needed=False),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cfg.update({
            'shuffle': False, 
            'drop_last': False,
        })
    dataset = WIDER(root, transform=transform)
    dataloader = DataLoader(dataset, **cfg)
    return dataloader


if __name__ == '__main__':
    # torch.manual_seed(9)
    transform = Compose([
        ToTensor(),
        ColorJitter(0.2, 0.2, 0.2, 0.2),
        RandomHorizontalFlip(p=0.5),
        RandomCrop((240, 360), padding=(64, 96), fill=127, pad_if_needed=False),
        # Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    ])
    # transform = None
    dataset = WIDER('/home/v-renjiechen/datasets/wider_yolo/WIDER_val', transform=transform)
    # print(len(dataset))
    img, lbl = dataset[0]

    def plot_bbox_cv2(im, bboxes, line_thickness=1):
        assert isinstance(im, np.ndarray)
        H, W, _ = im.shape
        draw = im.copy()
        for box in bboxes:
            x, y, w, h = round(box[0] * W), round(box[1]*H), round(box[2] * W), round(box[3] * H)
            cv2.rectangle(draw, (x-w//2, y-h//2), (x+w-w//2, y+h-h//2), (0, 0, 225), thickness=line_thickness, lineType=cv2.LINE_AA)
        return draw
    cv2.imwrite('sample.jpg', plot_bbox_cv2((img.permute(1, 2, 0).numpy()*255).astype(np.uint8), lbl.numpy()))
    # cv2.imwrite('sample.jpg', plot_bbox_cv2(img, lbl))