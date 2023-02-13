from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import hard_negative_mining

class MultiboxLoss(nn.Module):
    """Implement SSD Multibox Loss.

    Basically, Multibox loss combines classification loss and Smooth L1 regression loss.
    """
    def __init__(self, 
        num_classes: int, 
        neg_pos_ratio: Union[int, float], 
        center_variance: Union[int, float], 
        size_variance: Union[int, float]
    ):
        super(MultiboxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.num_classes = num_classes

    def forward(self, 
        confidence: torch.Tensor, 
        predicted_locations: torch.Tensor, 
        labels: torch.Tensor, 
        gt_locations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        mask = hard_negative_mining(confidence, labels, self.neg_pos_ratio)
        classification_loss = F.cross_entropy(
            confidence[mask].reshape(-1, self.num_classes), 
            labels[mask], 
            reduction='sum'
        )

        pos_mask = labels > 0
        smooth_l1_loss = F.smooth_l1_loss(
            predicted_locations[pos_mask].reshape(-1, 4), 
            gt_locations[pos_mask].reshape(-1, 4), 
            reduction='sum'
        )  # smooth_l1_loss
        num_pos = pos_mask.sum().item()
        return smooth_l1_loss / num_pos, classification_loss / num_pos