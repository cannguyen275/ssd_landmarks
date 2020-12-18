import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import box_processing as box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        """
            focusing is parameter that can adjust the rate at which easy
            examples are down-weighted.
            alpha may be set by inverse class frequency or treated as a hyper-param
            If you don't want to balance factor, set alpha to 1
            If you don't want to focusing factor, set gamma to 1 
            which is same as normal cross entropy loss
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.neg_pos_ratio = 7
        self.use_CrossEntropy = False

    def forward(self, conf_preds, loc_preds, land_preds, conf_targets, loc_targets, land_targets, ):
        """
            Args:
                predictions (tuple): (conf_preds, loc_preds)
                    conf_preds shape: [batch, n_anchors, num_cls]
                    loc_preds shape: [batch, n_anchors, 4]
                    land_preds shape: [batch, n_anchors, 10]
                targets (tensor): (conf_targets, loc_targets)
                    conf_targets shape: [batch, n_anchors]
                    loc_targets shape: [batch, n_anchors, 4]
                    land_targets shape: [batch, n_anchors, 10]
        """

        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        ############# Landmark Loss part ##############
        # zeros = torch.tensor(0).cuda()
        pos_land = conf_targets > 0  # ignore background and images without landmark 
        num_pos_landm = pos_land.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx_land = pos_land.unsqueeze(pos_land.dim()).expand_as(land_preds)
        land_p = land_preds[pos_idx_land].view(-1, 10)
        land_t = land_targets[pos_idx_land].view(-1, 10)
        land_loss = F.smooth_l1_loss(land_p, land_t, reduction='sum')

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos = conf_targets != 0
        conf_targets[pos] = 1

        ############# Localization Loss part ##############
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        loc_p = loc_preds[pos_idx].view(-1, 4)
        loc_t = loc_targets[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        ############### Confidence Loss part ###############
        if not self.use_CrossEntropy:
            # focal loss implementation(2)
            pos_cls = conf_targets > -1
            mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
            conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
            p_t_log = -F.cross_entropy(conf_p, conf_targets[pos_cls].type(torch.long), reduction='sum')
            p_t = torch.exp(p_t_log)

            # This is focal loss presented in the paper eq(5)
            conf_loss = -self.alpha * ((1 - p_t) ** self.gamma * p_t_log)
        else:
            num_classes = conf_preds.size(2)
            with torch.no_grad():
                # derived from cross_entropy=sum(log(p))
                loss = -F.log_softmax(conf_preds, dim=2)[:, :, 0]
                mask = box_utils.hard_negative_mining(loss, conf_targets, self.neg_pos_ratio)

            confidence = conf_preds[mask, :]
            conf_loss = F.cross_entropy(confidence.reshape(-1, num_classes), conf_targets[mask].type(torch.long),
                                        size_average=False)

        num_pos = pos.long().sum(1, keepdim=True)

        N = max(num_pos.data.sum(),
                1)  # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes
        conf_loss /= N  # exclude number of background?
        loc_loss /= N
        land_loss /= N1

        return loc_loss, conf_loss, land_loss

    @staticmethod
    def one_hot(x, n):
        y = torch.eye(n)
        return y[x]
