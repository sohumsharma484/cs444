import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detection_utils import compute_bbox_targets
from torch.autograd import Variable

class LossFunc(nn.Module):

    def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes):

        device = classifications.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []       

        anchor = anchors[0, :, :]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            targets_cls = gt_clss[j, :, :]
            targets_bbox = gt_bboxes[j, :, :]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            positive_indices = (targets_cls > 0).view(-1)
            non_negative_indices = (targets_cls >= 0).view(-1)
            num_positive_anchors = positive_indices.sum()

            if num_positive_anchors == 0:
                bce = -(torch.log(1.0 - classification))
                cls_loss = bce
                alpha = 0.25
                gamma = 2.0    
                p_t = torch.exp(-bce)
                alpha_tensor = (1 - alpha) #+ targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
                f_loss = alpha_tensor * (1 - p_t) ** gamma * bce
                cls_loss = f_loss
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().to(device))
                continue

            
            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.to(device)
            targets[non_negative_indices, :] = 0
            targets[positive_indices, targets_cls[positive_indices] - 1] = 1

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = bce

            alpha = 0.25
            gamma = 2.0    
            p_t = torch.exp(-bce)
            alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
            f_loss = alpha_tensor * (1 - p_t) ** gamma * bce
            cls_loss = f_loss
            

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            targets_bbox = targets_bbox[positive_indices, :]
            bbox_reg_target = compute_bbox_targets(anchor[positive_indices, :].reshape(-1,4), targets_bbox.reshape(-1,4))
            targets = bbox_reg_target.to(device)
            regression_diff = torch.abs(targets - regression[positive_indices, :])
            regression_losses.append(regression_diff.mean())


        # ce_loss = F.cross_entropy(classifications, gt_clss, reduction="none")
        # pt = torch.exp(-ce_loss)
        # focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # self.alpha = 0.25
        # self.gamma = 2.0
        # print(targets.shape)
        # print(gt_clss.shape)
        # ce_loss = torch.nn.CrossEntropyLoss(classifications, gt_clss, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        # pt = torch.exp(-ce_loss)
        # focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() 
        # return focal_loss
        # print(classifications.shape)
        # print(gt_clss.reshape((gt_clss.shape[0], -1)).shape)
        # print(gt_clss.reshape((gt_clss.shape[0], -1)).unique())
        # cls_targets = gt_clss.reshape((gt_clss.shape[0], -1))
        # cls_preds = classifications
        # pos_neg = cls_targets > -1  # exclude ignored anchors
        # mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        # masked_cls_preds = cls_preds[mask].view(-1,10)

        # y = torch.eye(11)  # [D,D]
        # target = y[cls_targets[pos_neg]]            # [N,D]
        # print("class: ", classification.shape, "target: ", old_targets.shape)
        # ce_loss = F.cross_entropy(classification, old_targets, reduction='none')
        # pt = torch.exp(-ce_loss)
        # focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # focal_loss = focal_loss.mean()
        # print(focal_loss)
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)



def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).to(self.device)  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        print(y.shape)
        grid = torch.eye(1 + self.num_classes)  
        t = grid[y.data.cpu()]  
        print(t.shape)
        test = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        assert (test == t).all()
        t = t[:,1:]
        t = Variable(t).to(self.device)

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.mean()

    # def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
    def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        loc_preds = regressions
        loc_targets = gt_bboxes
        cls_preds = classifications
        cls_targets = gt_clss.reshape((gt_clss.shape[0], -1))

        print(cls_targets.size())

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item()/num_pos, cls_loss.item()/num_pos), end=' | ')
        # loss = (loc_loss+cls_loss)/num_pos
        return cls_loss, loc_loss/num_pos