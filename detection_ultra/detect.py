"""
This code is used to batch detect images in a folder.
"""

import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from utils import generate_priors, xywh2xyxy, nms, convert_locations_to_boxes
from models import build_model

class Predictor:
    def __init__(self, net, priors, center_variance, size_variance, size, mean=0.5, std=1.0, nms_method=None, iou_threshold=0.3, filter_threshold=0.01, candidate_size=200, sigma=0.5):
        self.net = net
        self.priors = priors
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.mean = mean
        self.std = std
        self.size = size

        self.sigma = sigma
        self.device = next(net.parameters()).device

    def pre_process(self, img: np.ndarray):
        w, h = self.size
        H, W, _ = img.shape
        top_bottom = 0
        left_right = 0
        if W/w > H/h:
            top_bottom = round(W / w * h) - H
        else:
            left_right = round(H / h * w) - W
        # pad
        img = cv2.copyMakeBorder(img, top_bottom//2, top_bottom-top_bottom//2, left_right//2, left_right-left_right//2, cv2.BORDER_CONSTANT, value=(0 ,0 ,0))
        img = cv2.resize(img, (w, h))
        img = (img-self.mean)/self.std
        inputs = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().float()
        return inputs.unsqueeze(dim=0).to(self.device), top_bottom, left_right

    @torch.no_grad()
    def inference(self, img, top_k=-1, prob_threshold=None):
        H, W, _ = img.shape
        inputs, top_bottom, left_right = self.pre_process(img.copy())
        
        confidences, locations = self.net(inputs)
        scores = F.softmax(confidences, dim=2)
        boxes = convert_locations_to_boxes(locations, self.priors, self.center_variance, self.size_variance)
        boxes = xywh2xyxy(boxes)

        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.cpu()
        scores = scores.cpu()
        picked_box_probs = []

        probs = scores[:, 1]
        mask = probs > prob_threshold
        if mask.sum().item() > 0:
            box_probs = nms(
                torch.cat([boxes[mask], probs[mask].reshape(-1, 1)], dim=1), 
                self.nms_method,
                score_threshold=prob_threshold,
                iou_threshold=self.iou_threshold,
                sigma=self.sigma,
                top_k=top_k,
                candidate_size=self.candidate_size
            )
            picked_box_probs.append(box_probs)
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, [0, 2]] *= (W + left_right)
            picked_box_probs[:, [1, 3]] *= (H + top_bottom)
            picked_box_probs[:, [0, 2]] -= left_right // 2
            picked_box_probs[:, [1, 3]] -= top_bottom // 2
            return picked_box_probs[:, :4].long().numpy(), picked_box_probs[:, 4].float().numpy()
        else:
            return np.array([]), np.array([])




if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='detect imgs')
    parser.add_argument('--arch', default="slim", type=str, help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--weight', default="weight.pth", type=str, help='pretrained model weight like *.pth')
    parser.add_argument('--input_size', default=1280, type=int, help='define network input size,default optional value 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.6, type=float, help='score threshold')
    parser.add_argument('--candidate_size', default=1500, type=int, help='nms candidate size')
    parser.add_argument('--path', default="imgs/2.jpg", type=str, help='imgs dir')
    parser.add_argument('--cuda', default=True, type=bool, help='cuda or cpu')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2

    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    feature_map_w_h_list = [[40, 20, 10, 5], [30, 15, 8, 4]]  # default feature map size

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
    image_size = img_size_dict[args.input_size]

    feature_map_w_h_list = feature_map_w_h_list_dict[args.input_size]

    shrinkage_list = [
        [ image_size[i] / feature_map_w_h_list[i][k] for k in range(len(feature_map_w_h_list[i]))] 
        for i in range(len(image_size)) 
    ]
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    priors = priors.to(device)

    model_path = 'weights-rfb.pth'
    model = build_model(2, args.arch).to(device)
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    predictor = Predictor(
        model, priors, center_variance, size_variance,
        image_size, (127, 127, 127), 128,
        nms_method=None,
        iou_threshold=iou_threshold,
        candidate_size=args.candidate_size,
        sigma=0.5,
    )

    sum = 0

    img_path = args.path
    orig_image = cv2.imread(img_path)
    # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, probs = predictor.inference(orig_image, args.candidate_size / 2, args.threshold)

    sum += boxes.shape[0]
    for i in range(boxes.shape[0]):
        box = boxes[i]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        # label = f"{probs[i]:.2f}"
        # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(orig_image, str(boxes.shape[0]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite("detected.jpg", orig_image)
    print(f"Found {len(probs)} faces.")
