import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader


import torchvision.transforms as T

from data import FairFace

def build_model(backbone='resnet18', out_channels=10, pretrained=True):
    assert backbone in ['resnet18', 'resnet34']
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(num_classes=1000, weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, out_channels)
    return model

def main(opts):
    # hyp = opts.hyp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224, padding=32, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = FairFace('/home/v-renjiechen/datasets/fairface', split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=FairFace.collate_fn, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)

    
    model = build_model('resnet18')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    l2_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    model.train()

    for epoch in range(20):
        with tqdm(dataloader, total=len(dataloader), desc=f'Train{epoch:2d}', ncols=120) as t:
            acc_age = 0
            acc_gender = 0
            acc_race = 0
            for images, targets in t:
                images, targets = images.to(device), targets.to(device)
                targets_age, targets_gender, targets_race = targets.chunk(3, dim=-1)
                targets_gender, targets_race = targets_gender.squeeze(dim=-1).long(), targets_race.squeeze(dim=-1).long()
                logits = model(images)
                logits_age, logits_gender, logits_race = torch.split(logits, split_size_or_sections=(1, 2, 7), dim=-1)
                loss = 2*criterion(logits_gender, targets_gender) + 2 * l2_criterion(torch.sigmoid(logits_age), targets_age) + criterion(logits_race, targets_race)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                acc_age = 0.9 * acc_age + 0.1 * (torch.sigmoid(logits_age.detach()) - targets_age).abs().mean().item()
                acc_gender = 0.9 * acc_gender + 0.1 * (logits_gender.argmax(dim=-1) == targets_gender).sum().item() / targets_gender.size(0)
                acc_race = 0.9 * acc_race + 0.1 * (logits_race.argmax(dim=-1) == targets_race).sum().item() / targets_race.size(0)
                t.set_postfix({
                    'loss': f'{loss.item():.3f}', 
                    'age': f'{acc_age:.2f}', 
                    'gender': f'{acc_gender:.2%}', 
                    'race': f'{acc_race:.2%}'
                })
            scheduler.step()
        torch.save(model.state_dict(), 'weights.pth')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Face Classification')
    parser.add_argument('--cfg', default='mobile.yaml', type=str, help='setup config like *.yaml')
    parser.add_argument('--resume', default=None, help='resume net for retraining')
    args = parser.parse_args()

    main(args)