import os
import math
import yaml
import random
import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data import WIDER
from augment import preproc
from loss import MultiBoxLoss
from prior_box import PriorBox
from retinaface import RetinaFace

from collections import OrderedDict

cudnn.benchmark = True
# cudnn.deterministic = True

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

is_debug = cudnn.deterministic

def main(args):
    assert args.cfg and os.path.exists(args.cfg)
    with open(args.cfg, 'r') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rgb_mean = (104, 117, 123) # bgr order
    imgsz = hyp['image_size']
    batch_size = hyp['batch_size']
    epochs = hyp['epochs']
    num_workers = hyp['num_workers']
    lr = hyp['lr']

    save_folder = args.save_folder

    dataset = WIDER(hyp['datapath'], preproc(imgsz, rgb_mean))
    dataloader = DataLoader(dataset, batch_size, shuffle=not is_debug, num_workers=num_workers, collate_fn=WIDER.collate_fn, drop_last=True, pin_memory=True)

    num_batches = len(dataset) // batch_size
    total_iters = epochs * num_batches
    epoch = 0

    if hyp['backbone'] == 'mobilenet0.25':
        from modules import MobileNetV1
        backbone = MobileNetV1()
        if True or hyp['pretrain']:
            checkpoint = torch.load("../weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            # load params
            backbone.load_state_dict(new_state_dict)
    elif hyp['backbone'] == 'Resnet50':
        import torchvision
        backbone = torchvision.models.resnet50(pretrained=hyp['weights'])

    model = RetinaFace(backbone, hyp['in_channels'], hyp['out_channels'], hyp['return_layers'])

    if args.resume and isinstance(args.resume, str):
        print('Loading resume network...')
        state_dict = torch.load(args.resume)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # epoch = resume_epoch

    model = model.to(device)
    criterion = MultiBoxLoss(overlap_thresh=0.35, neg_pos=7)

    with torch.no_grad():
        priorbox = PriorBox(hyp['min_sizes'], hyp['strides'], (imgsz, imgsz), hyp['clip'])
        priors = priorbox.forward().to(device)
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hyp['milestones'], gamma=0.1, last_epoch=- 1)
    lr = scheduler.get_last_lr()[0]

    print(f"Num batches: {num_batches}")
    start_iter = epoch * num_batches
    pbar = tqdm(range(start_iter, total_iters), total=total_iters, desc=f"Train(0)", ncols=100, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt:3s} [{elapsed}<{remaining}{postfix}]", disable=is_debug)

    
    model.train()
    dataiterator = iter(dataloader)
    for iteration in pbar:
        # load train data
        try:
            images, targets = next(dataiterator)
        except StopIteration:
            torch.save(model.state_dict(), os.path.join(save_folder, "mobilenet3.pth"))
            dataiterator = iter(dataloader)
            epoch += 1
            pbar.set_description(f'Train({epoch:2d})')
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            images, targets = next(dataiterator)
        except Exception as e:
            print(e)
            exit()
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = model(images)
        # backprop
        loss_box, loss_obj, loss_ldm = criterion(out, priors, targets)
        loss = hyp['box_weight'] * loss_box + loss_obj + loss_ldm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f'Box: {loss_box.item():.3f} Obj: {loss_obj.item():.3f} Ldm: {loss_ldm.item():.3f} | LR: {lr:.5f}')

    pbar.close()
    torch.save(model.state_dict(), os.path.join(save_folder, "mobilenet_finalv2.pth"))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--save_folder', default='../weights/', help='Location to save checkpoint models')
    parser.add_argument('--cfg', default='configs/mobile.yaml', type=str, help='setup config like *.yaml')
    parser.add_argument('--resume', default=None, help='resume net for retraining')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    main(args)
