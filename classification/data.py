import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class FairFace(Dataset):
    def __init__(self, root, split='train', transform=None):
        super(FairFace, self).__init__()
        assert split in ['train', 'val']
        assert transform is not None
        self.root = root
        self.split = split
        self.transform = transform

        df = pd.read_csv(
            os.path.join(root, f'fairface_label_{split}.csv'), 
            sep=',', header=0, index_col='file', usecols=['file', 'gender', 'race', 'age'],
        )
        # pd.unique(df.race)
        df.gender = df.gender.map(
            {n: i for i, n in enumerate(['Male', 'Female'])}
        )
        # df.age = df.age.map(
        #     {n: i for i, n in enumerate(['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70'])}
        # )
        df.age = df.age.map(
            {'0-2': 0.05, '3-9': 0.05, '10-19': 0.15, '20-29': 0.25, '30-39': 0.35, 
            '40-49': 0.45, '50-59': 0.55, '60-69': 0.65, 
            'more than 70': 0.8}
        )
        df.race = df.race.map(
            {n: i for i, n in enumerate(['East Asian', 'White', 'Latino_Hispanic', 'Southeast Asian',
            'Black', 'Indian', 'Middle Eastern'])}
        )

        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        k = self.df.index[index]
        img = Image.open(os.path.join(self.root, k))
        lbl = self.df.loc[k].values
        img = self.transform(img)
        lbl = torch.from_numpy(lbl).float()

        return img, lbl

    @staticmethod
    def collate_fn(batch):
        imgs, tars = list(zip(*batch))

        # Resize images to input shape
        imgs = torch.stack(imgs, dim=0)
        tars = torch.stack(tars, dim=0)
        return imgs, tars