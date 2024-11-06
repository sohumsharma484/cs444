import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0, 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
    gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
    Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
    and the gt_bbox should again arbitrarilarly be set to all zeros).
    Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
    You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but in order to make your code efficient 
    during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
    This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
    """
    # TODO(student): Complete this function
    device = anchor.device
    gt_clss = torch.zeros(anchor.shape[0], anchor.shape[1], 1).to(device)
    gt_bboxes = torch.zeros(anchor.shape[0], anchor.shape[1], 4).to(device)
    zeros = torch.tensor([0, 0, 0, 0]).float().to(device)
    for i in range(anchor.shape[0]):
        ious = compute_bbox_iou(anchor[0], bbox[0])
        max_ious, max_indices = torch.max(ious, dim=1)
        gt_clss[i][max_ious < 0.5] = -1
        gt_bboxes[i][max_ious < 0.5] = zeros
        gt_clss[i][max_ious < 0.4] = 0
        gt_bboxes[i][max_ious < 0.4] = zeros
        gt_clss[i][max_ious >= 0.5] = cls[i][max_indices[max_ious >= 0.5]].float().to(device)
        gt_bboxes[i][max_ious >= 0.5] = bbox[i][max_indices[max_ious >= 0.5]].to(device)

    return gt_clss.to(torch.int), gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth object classes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we compute are with respect to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
    We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
    our regression targets more invariant to the size of the original anchor box. So, this means that:
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_w = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle varying sizees of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
       This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
       our regression target would be different. Therefore our formula changes to delta_w = log(gt_bbox_width / anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
       high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
    """
    # TODO(student): Complete this function
    delta_x = ((gt_bboxes[:,2] + gt_bboxes[:,0]) / 2 - (anchors[:,2] + anchors[:,0]) / 2) / (anchors[:,2] - anchors[:,0])
    delta_y = ((gt_bboxes[:,3] + gt_bboxes[:,1]) / 2 - (anchors[:,3] + anchors[:,1]) / 2) / (anchors[:,3] - anchors[:,1])
    delta_w = torch.log(torch.clamp(gt_bboxes[:,2] - gt_bboxes[:,0], min=1) /(anchors[:,2] - anchors[:,0]))
    delta_h = torch.log(torch.clamp(gt_bboxes[:,3] - gt_bboxes[:,1], min=1) /(anchors[:,3] - anchors[:,1]))
    return torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1).to(anchors.device)

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # TODO(student): Complete this function
    new_boxes = boxes.clone()
    width = boxes[:,2] - boxes[:,0]
    center_x = deltas[:,0] * width + (boxes[:,0] + boxes[:,2]) / 2
    new_width = torch.exp(deltas[:,2]) * width
    new_boxes[:,0] = center_x - new_width / 2
    new_boxes[:,2] = center_x + new_width / 2

    height = boxes[:,3] - boxes[:,1]
    center_y = deltas[:,1] * height + (boxes[:,1] + boxes[:,3]) / 2
    new_height = torch.exp(deltas[:,3]) * height
    new_boxes[:,1] = center_y - new_height / 2
    new_boxes[:,3] = center_y + new_height / 2
    return new_boxes.to(boxes.device)

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is used to prevent having many boxes that overlap each other. To do this, if multiple boxes overlap each other beyond a
    threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
    first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
    best bbox(the one with the highest score), It will be choosen over all overlapping boxes. Therefore, you should add this bbox to your final 
    resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

    make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
    """
    # TODO(student): Complete this function
    ious = compute_bbox_iou(bboxes, bboxes)
    keep = []
    _, indices = scores.sort(descending=True)
    indices = indices.tolist()
    for idx in indices:
        if len(keep) == 0:
            keep.append(idx)
            continue
        iou = ious[idx][keep]
        if (iou < threshold).all():
            keep.append(idx)

    return torch.tensor(keep).long().to(bboxes.device)

# import torch
# import numpy as np
# import random

# def set_seed(seed):
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)

    
# def get_detections(outs):
#     B, BB, _, _ = outs[0][0].shape
#     _, A, _, _ = outs[0][2].shape
#     A = A // 4
#     num_classes = BB // A
    
#     pred_bboxes, pred_clss, anchors = [], [], []
#     for pred_cls, pred_bbox, anchor in outs:
#         # Get all the anchors, pred and bboxes
#         H, W = pred_cls.shape[-2:]
#         pred_cls = pred_cls.reshape(B, A, -1, H, W)
#         pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

#         pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
#         pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

#         anchor = anchor.reshape(B, A, -1, H, W)
#         anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
#     pred_clss = torch.cat(pred_clss, dim=1)
#     pred_bboxes = torch.cat(pred_bboxes, dim=1)
#     anchors = torch.cat(anchors, dim=1)
#     return pred_clss, pred_bboxes, anchors

# def compute_bbox_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, dim=1):
#     """
#     Args:
#         bbox1: (B, N, 4) tensor of (x1, y1, x2, y2)
#         bbox2: (B, M, 4) tensor of (x1, y1, x2, y2)
#     Returns
#         iou: (B, N, M) tensor of IoU values
#     """
#     has_batch = len(bbox1.shape) > 2
#     N = bbox1.shape[-2] if len(bbox1.shape) > 1 else 1
#     M = bbox2.shape[-2] if len(bbox2.shape) > 1 else 1
#     orig_shape2 = bbox2.shape
#     bbox1 = bbox1.reshape(-1, N, 1, 4)
#     bbox2 = bbox2.reshape(-1, 1, M, 4)
#     max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
#     min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
#     max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
#     min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
#     intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
#     area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
#     area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
#     iou = intersection / (area1 + area2 - intersection)
#     if len(orig_shape2) == 1:
#         iou = iou.reshape(N)
#     if not has_batch:
#         iou = iou.squeeze(0)
#     return iou

# def compute_targets(anchor: torch.Tensor, cls: torch.Tensor, bbox: torch.Tensor):
#     """
#     Args:
#         anchor: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
#         cls: groundtruth object classes of shape (B, number of objects in the image, 1)
#         bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
#     Returns:
#         gt_clss: groundtruth class targets of shape (B, A, 1)
#         gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
#     Hint: remember if the max_iou for that bounding box is between [0, 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
#     gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
#     Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
#     and the gt_bbox should again arbitrarilarly be set to all zeros).
#     Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
#     Hint: use torch.max to get both the max iou and the index of the max iou.

#     Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

#     Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
#     You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

#     VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but in order to make your code efficient 
#     during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
#     you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
#     This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
#     """
#     # TODO(student): Complete this function
#     B, A, _ = anchor.shape
#     _, N, _ = cls.shape
#     ious = compute_bbox_iou(anchor, bbox)
#     assert ious.shape == (B, A, N), f"Expected ious to have shape {(B, A, N)} but got {ious.shape}"
#     # print(anchor.shape, bbox.shape, cls.shape, ious.shape)
#     max_ious, max_indices = torch.max(ious, dim=-1)
#     assert max_ious.shape == (B, A), f"Expected max_ious to have shape {(B, A)} but got {max_ious.shape}"
#     above = max_ious >= 0.5
#     assert above.shape == (B, A), f"Expected above to have shape {(B, A)} but got {above.shape}"
#     above_idx = max_indices[above]
#     # print(above_idx)
#     gt_clss = torch.zeros((B, A, 1)).to(torch.int).to(anchor.device)
#     gt_bboxes = torch.zeros((B, A, 4)).to(anchor.device)
#     # print(max_ious.shape, max_indices.shape, above.shape, above_idx.shape)
#     # print(gt_clss.shape, gt_bboxes.shape)
#     # print(gt_clss[above].shape, cls[:, above_idx, :].shape)

#     batch_indices = torch.arange(B).unsqueeze(1).expand(B, A).reshape(-1).to(anchor.device)
#     above_batch_indices = batch_indices[above.reshape(-1)]

#     gt_clss[max_ious >= 0.4] = -1
#     gt_clss[above] = cls[above_batch_indices, above_idx, :].to(torch.int)

#     gt_bboxes[above] = bbox[above_batch_indices, above_idx, :]

#     return gt_clss.to(anchor.device), gt_bboxes.to(anchor.device)

# def compute_bbox_targets(anchors: torch.Tensor, gt_bboxes: torch.Tensor):
#     """
#     Args:
#         anchors: anchors of shape (A, 4)
#         gt_bboxes: groundtruth object classes of shape (A, 4)
#     Returns:
#         bbox_reg_target: regression offset of shape (A, 4)
    
#     Remember that the delta_x and delta_y we compute are with respect to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
#     We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
#     our regression targets more invariant to the size of the original anchor box. So, this means that:
#     delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

#     When computing delta_w and delta_h, there are a few things to note.
#     1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_w = gt_bbox_width / anchor_width
#     2. Logarithm: In order to make our regression targets better handle varying sizes of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
#        This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
#        our regression target would be different. Therefore our formula changes to delta_w = log(gt_bbox_width / anchor_width)
#     3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
#        high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
#        delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
#     """
#     # TODO(student): Complete this function
#     w = anchors[:, 2] - anchors[:, 0]
#     h = anchors[:, 3] - anchors[:, 1]
#     gw = gt_bboxes[:, 2] - gt_bboxes[:, 0]
#     gh = gt_bboxes[:, 3] - gt_bboxes[:, 1]
#     x = anchors[:, 0] + w / 2
#     y = anchors[:, 1] + h / 2
#     gx = gt_bboxes[:, 0] + gw / 2
#     gy = gt_bboxes[:, 1] + gh / 2
#     delta_x = (gx - x) / w
#     delta_y = (gy - y) / h
#     delta_w = torch.log(torch.clamp(gw, min=1) / w)
#     delta_h = torch.log(torch.clamp(gh, min=1) / h)

#     return torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

# def apply_bbox_deltas(boxes, deltas):
#     """
#     Args:
#         boxes: (N, 4) tensor of (x1, y1, x2, y2)
#         deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
#     Returns
#         boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
#     """
#     xy = (boxes[:, (0,1)] + boxes[:, (2,3)]) / 2
#     wh = boxes[:, (2,3)] - boxes[:, (0,1)]
#     nxy = xy + deltas[:, (0,1)] * wh
#     nwh = wh * torch.exp(deltas[:, (2,3)])
#     new_boxes = torch.cat([nxy - nwh / 2, nxy + nwh / 2], dim=-1)
#     return new_boxes

# def nms(bboxes: torch.Tensor, scores: torch.Tensor, threshold=0.5):
#     """
#     Args:
#         bboxes: (N, 4) tensor of (x1, y1, x2, y2)
#         scores: (N,) tensor of scores
#         threshold: float of IoU threshold to use for NMS
#     Returns:
#         keep: (K,) tensor of indices to keep
    
#     Remember that nms is used to prevent having many boxes that overlap each other. To do this, if multiple boxes overlap each other beyond a
#     threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
#     first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
#     best bbox(the one with the highest score), It will be chosen over all overlapping boxes. Therefore, you should add this bbox to your final
#     resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

#     make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
#     """
#     # TODO(student): Complete this function
#     N, = scores.shape
#     # I documented all of this because I am confused as to what I am writing


#     # Gets indices of bboxes sorted by score in dimension N
#     to_analyze = torch.argsort(scores, descending=True)
#     # assert torch.equal(to_analyze, torch.arange(N)[to_analyze])  # To make sure I know how indices work

#     # Computes NxN matrix of ious. Matrix is symmetric. I can figure out how to make it do half of the operations later
#     # ious = compute_bbox_iou(bboxes, bboxes).reshape(N, N)
#     # assert torch.equal(ious, ious.T)  # To make sure it's symmetric

#     # Filters out ious that are below threshold. Returns a boolean matrix. Should still be symmetric
#     # ious = ious > threshold
#     # assert torch.equal(ious, ious.T)  # To make sure it's still symmetric

#     # Now I need to find the bbox with the highest score in each row. I can do this by sorting the scores in each row
#     # by the indices found before. I first rearrange them like I did before, then I sort them by the indices in each row.
#     # Since each element is a boolean, the highest score will be the first one that is not the constant.
#     # dont_keep = ious[:, to_analyze].to(torch.int8).argsort(dim=1, decreasing=True)[:, 1:]

#     # keep = torch.unique(to_analyze[ious])
#     # TODO: do a don't keep
#     # return keep




#     dont_keep = torch.zeros(N, dtype=torch.bool)
#     sorted_bboxes = bboxes[to_analyze]
#     for i in range(N-1):
#         if dont_keep[i]:
#             continue
#         ious = compute_bbox_iou(sorted_bboxes[i].unsqueeze(0), sorted_bboxes[i+1:]).ravel() > threshold
#         dont_keep[i+1:][ious] = True

#     return to_analyze[~dont_keep].to(torch.int)