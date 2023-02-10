import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_targets

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, overlap_thresh, neg_pos):
        super(MultiBoxLoss, self).__init__()
        self.variance = [0.1, 0.2]
        self.num_classes = 2
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        # self.background_label = bkg_label
        # self.encode_target = encode_target
        # self.use_prior_for_matching = prior_for_matching
        # self.do_neg_mining = neg_mining
        
        # self.neg_overlap = neg_overlap
        

    def forward(self, preds, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # build targets. 
        # num_priors = num_all_cells * num_anchors
        # pbox/tbox: [batch_size, num_priors, 4].
        # pldm/tldm: [batch_size, num_priors, 10].
        # pobj: [batch_size, num_priors, 2], tobj: [batch_size, num_priors]. tobj in {-1, +1}
        pbox, pobj, pldm = preds
        tbox, tobj, tldm = build_targets(targets, self.threshold, priors, self.variance)
        batch_size = pbox.size(0)

        zeros = 0
        # landm Loss (Smooth L1)
        pos = (tobj > zeros)
        N1 = max(pos.sum().float(), 1)
        # [batch_size, num_priors, 10]
        pos_mask = pos.unsqueeze(dim=pos.ndim).expand_as(pldm)
        loss_ldm = F.smooth_l1_loss(
            pldm[pos_mask].view(-1, 10), 
            tldm[pos_mask].view(-1, 10),
            reduction='sum'
        )

        # pos means a bbox.
        pos = (tobj != zeros)
        # -1 and 1 to 1. no landmark but a bbox
        tobj[pos] = 1

        # Localization Loss (Smooth L1)
        # [batch_size, num_priors, 4]
        pos_mask = pos.unsqueeze(dim=pos.ndim).expand_as(pbox)
        loss_box = F.smooth_l1_loss(
            pbox[pos_mask].view(-1, 4), 
            tbox[pos_mask].view(-1, 4),
            reduction='sum'
        )

        with torch.no_grad():
            # Compute max conf across batch for hard negative mining
            # [batch_size, num_priors, 2] -> [batch_size*num_priors, 2]
            pobj_logits = pobj.view(-1, self.num_classes)
            # loss = log_sum_exp(batch_conf) - batch_conf.gather(1, tobj.view(-1, 1))
            # [batch_size*num_priors, 1]
            loss = torch.logsumexp(pobj_logits, dim=1, keepdim=True) - torch.gather(pobj_logits, dim=1, index=tobj.view(-1, 1))
            # Hard Negative Mining
            loss[pos.view(-1, 1)] = 0 # filter out pos boxes for now
            # [batch_size, num_priors]
            loss = loss.view(batch_size, -1)
            # [batch_size, num_priors]. get the index of element in sorted sequence.
            idx_rank = loss.argsort(dim=1, descending=True).argsort(dim=1)
            # [batch_size, 1]
            num_pos = pos.sum(dim=1, keepdim=True)
            # [batch_size, num_priors]. control the negative sample ratio.
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1)-1).expand_as(idx_rank)
            neg = idx_rank < num_neg

            # Confidence Loss Including Positive and Negative Examples
            pos_or_neg = (pos | neg)

        loss_obj = F.cross_entropy(
            pobj[pos_or_neg.unsqueeze(dim=2).expand_as(pobj)].view(-1, self.num_classes), 
            tobj[pos_or_neg], reduction='sum'
        )
 
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N  = max(num_pos.sum().float(), 1)

        loss_box /= N
        loss_obj /= N
        loss_ldm /= N1

        return loss_box, loss_obj, loss_ldm
