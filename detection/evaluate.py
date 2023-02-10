import os
import yaml

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

from prior_box import PriorBox
from utils import py_cpu_nms, decode, decode_landm
import cv2
from retinaface import RetinaFace


@torch.no_grad()
def main(opts):
    assert opts.cfg and os.path.exists(opts.cfg)
    with open(opts.cfg, 'r') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    
    device = torch.device('cuda' if torch.cuda.is_available() and opts.cuda else 'cpu')

    if hyp['backbone'] == 'mobilenet0.25':
        from modules import MobileNetV1
        backbone = MobileNetV1()
    elif hyp['backbone'] == 'Resnet50':
        import torchvision
        backbone = torchvision.models.resnet50(pretrained=hyp['weights'])
    
    net = RetinaFace(backbone, hyp['in_channels'], hyp['out_channels'], hyp['return_layers'])
    net.load_state_dict(torch.load(opts.weight))
    net = net.to(device)
    net.eval()

    # testing dataset
    data_folder = opts.data_folder
    anno_file = os.path.join(os.path.dirname(data_folder), "wider_val.txt")
    with open(anno_file, 'r') as f:
        img_files = f.read().split()
    num_imgs = len(img_files)

    print(f'Total {num_imgs} files to be evaluated!')

    # testing scale
    target_size = 1600
    max_size = 2150
    rgb_mean = (104, 117, 123) # bgr order
    print("Start evaluate!")

    for i, img_file in enumerate(img_files[:3]):
        img_file = img_file.lstrip('/')
        im = cv2.imread(os.path.join(data_folder, img_file), cv2.IMREAD_COLOR)
        im = im.astype(np.float32)
        im_shape = im.shape[0:2]

        if opts.origin_size:
            resize = 1
        else:
            im_size_min = min(im_shape)
            im_size_max = max(im_shape)
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size
            if round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)

        if resize != 1:
            im = cv2.resize(im, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
        im_height, im_width = im.shape[:2]
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
        im -= rgb_mean
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).unsqueeze(dim=0)
        im = im.to(device)

        loc, conf, landms = net(im)  # forward pass
        conf = F.softmax(conf, dim=-1)

        priorbox = PriorBox(hyp['min_sizes'], hyp['strides'], (im_height, im_width), hyp['clip'])
        priors = priorbox().to(device)
        boxes = decode(loc.squeeze(dim=0), priors, hyp['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(dim=0)[:, 1].cpu().numpy()
        landms = decode_landm(landms.squeeze(dim=0), priors, hyp['variance'])
        scale1 = torch.Tensor([im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                               im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                               im.shape[3], im.shape[2]]).to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        for box in dets:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence # + " \n"
            print(line)

        # save_name = args.save_folder + img_file[:-4] + ".txt"
        # dirname = os.path.dirname(save_name)
        # if not os.path.isdir(dirname):
        #     os.makedirs(dirname)
        # with open(save_name, "w") as fd:
        #     bboxs = dets
        #     file_name = os.path.basename(save_name)[:-4] + "\n"
        #     bboxs_num = str(len(bboxs)) + "\n"
        #     fd.write(file_name)
        #     fd.write(bboxs_num)
        #     for box in bboxs:
        #         x = int(box[0])
        #         y = int(box[1])
        #         w = int(box[2]) - int(box[0])
        #         h = int(box[3]) - int(box[1])
        #         confidence = str(box[4])
        #         line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
        #         fd.write(line)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('--cfg', default='mobilenet.yaml', type=str, help='predefined config')
    parser.add_argument('--weight', default='./weights', required=True, help='pretrained weights.')
    parser.add_argument('--data_folder', default='./data/widerface/val/images/', type=str, help='dataset path')

    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use GPU inference')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')

    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()

    main(args)

"""
usage:
    $ python evaluate.py --weight /home/v-renjiechen/workspace/nnface/weights/mobilenet_final.pth --data_folder /home/v-renjiechen/data/widerface/val/images --cfg configs/mobile.yaml
"""