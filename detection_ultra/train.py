import os
import sys

import random
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True
# cudnn.deterministic = True

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

from models import build_model, MultiboxLoss
from data import build_loader
from utils import generate_priors, MatchPrior

def train(
    model: nn.Module, dataloader: DataLoader, 
    criterion: nn.Module, optimizer: optim.Optimizer, 
    scheduler: lr_scheduler._LRScheduler, 
    match_prior: MatchPrior,
    epoch: int, device: torch.device
):
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_clf_loss = 0.0
    # reg_loss: torch.Tensor = 0
    # clf_loss: torch.Tensor = 0
    lr = scheduler.get_last_lr()[0]
    with tqdm(dataloader, total=len(dataloader), desc=f'Train ({epoch:>3d}) lr={lr:.4f}', ncols=100, disable=False) as t:
        for it, (images, boxes) in enumerate(t):
            images = images.to(device)
            boxes, labels = list(zip(*[match_prior(b.to(device), torch.ones(b.size(0), dtype=torch.long, device=device)) for b in boxes]))
            boxes = torch.stack(boxes, dim=0)
            labels = torch.stack(labels, dim=0)

            confidence, locations = model(images)
            reg_loss, clf_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
            loss = reg_loss + clf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_clf_loss += clf_loss.item()
            t.set_postfix({'loss': f'{total_loss/(it+1):.3f}', 'reg': f'{total_reg_loss/(it+1):.3f}', 'clf': f'{total_clf_loss/(it+1):.3f}'})

    scheduler.step()
    return total_loss / (it+1), total_reg_loss / (it+1), total_clf_loss / (it+1)

def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() and opts.use_cuda else "cpu")

    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2
    neg_pos_ratio = 3
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    img_size_dict = {
        128: [128, 96],
        160: [160, 120],
        320: [320, 240],
        480: [480, 360],
        640: [640, 480],
        1280: [1280, 960]
    }
    feature_map_w_h_list_dict = {
        128: [[16, 8, 4, 2], [12, 6, 3, 2]],
        160: [[20, 10, 5, 3], [15, 8, 4, 2]],
        320: [[40, 20, 10, 5], [30, 15, 8, 4]],
        480: [[60, 30, 15, 8], [45, 23, 12, 6]],
        640: [[80, 40, 20, 10], [60, 30, 15, 8]],
        1280: [[160, 80, 40, 20], [120, 60, 30, 15]]
    }

    image_size = img_size_dict[opts.input_size]
    feature_map_w_h_list = feature_map_w_h_list_dict[opts.input_size]
    shrinkage_list = [
        [ image_size[i] / feature_map_w_h_list[i][k] for k in range(len(feature_map_w_h_list[i]))] 
        for i in range(len(image_size)) 
    ]

    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes).to(device)
    match_prior = MatchPrior(priors, center_variance, size_variance, opts.overlap_threshold)

    train_loader = build_loader(opts.train_dataset, 'train', opts.batch_size, opts.num_workers)
    num_classes = len(train_loader.dataset.class_names)

    # val_dataset = WIDER(opts.val_dataset, transform=val_transform, target_transform=target_transform)
    # val_loader = DataLoader(val_dataset, opts.batch_size, num_workers=opts.num_workers, shuffle=False, collate_fn=WIDER.collate_fn)

    
    last_epoch = -1
    model = build_model(num_classes, opts.arch).to(device)
    criterion = MultiboxLoss(num_classes, neg_pos_ratio, center_variance, size_variance)

    optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=args.gamma, last_epoch=last_epoch)

    for epoch in range(last_epoch + 1, opts.num_epochs):
        train(model, train_loader, criterion, optimizer, scheduler, match_prior, epoch, device)

        # if epoch % opts.validation_epochs == 0 or epoch == opts.num_epochs - 1:
        #     val_loss, val_reg_loss, val_clf_loss = valid(net, val_loader, criterion, device)
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 'weights-rfb.pth')
    torch.save(model.state_dict(), 'weights-rfb.pth')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train With Pytorch')

    parser.add_argument('--train_dataset', help='Dataset directory path')
    parser.add_argument('--val_dataset', help='Dataset directory path')
    parser.add_argument('--balance_data', action='store_true', help="Balance training data by down-sampling more frequent labels.")

    parser.add_argument('--arch', default="RFB", help="The network architecture ,optional(RFB , slim)")

    # Params for SGD
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer_type')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    

    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--weights', help='Pretrained base model')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str, help="Scheduler for SGD. It can one of multi-step and cosine")
    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', nargs="+", type=int, help="milestones for MultiStepLR")
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float, help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--input_size', default=320, type=int, help='define network input size,default optional value 128/160/320/480/640/1280')
    parser.add_argument('--batch_size', default=24, type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', default=200, type=int, help='the number epochs')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=5, type=int, help='the number epochs')
    parser.add_argument('--debug_steps', default=100, type=int, help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use CUDA to train model')

    parser.add_argument('--overlap_threshold', default=0.35, type=float, help='overlap_threshold')

    args = parser.parse_args()
    # print(args)
    main(args)