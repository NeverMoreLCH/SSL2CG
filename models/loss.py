import torch
import torch.nn.functional as F

def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy_with_logits(scores.masked_select(masks), 
                                            target_prob.masked_select(masks))
    return loss

def bce(scores, masks, targets, cfg, batch_weight=None):
    # scores = torch.sigmoid(scores) * masks

    # min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    # target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    # target_prob[target_prob > 0] += bias
    # target_prob[target_prob > 1] = 1
    # target_prob[target_prob < 0] = 0
    # loss = F.binary_cross_entropy(scores[:,:,:target_prob.shape[2]], target_prob, reduction='none') * masks[:,:,:target_prob.shape[2]]
    # loss = loss * batch_weight
    # loss_value = torch.sum(loss) / torch.sum(masks)
    # # loss_value = bce_rescale_loss(scores, masks, targets, cfg)
    if not batch_weight is None:
        raise NotImplementedError
    loss_value = bce_rescale_loss(scores, masks, targets, cfg)
    return loss_value * cfg.WEIGHT

def ssl(scores, masks, targets, cfg, batch_weight=None):
    scores = torch.sigmoid(scores)

    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0 # min_iou
    loss = scores * target_prob
    if not batch_weight is None:
        loss = loss * batch_weight[..., None, None]
    loss_value = loss.masked_select(masks).mean()
    return loss_value * cfg.WEIGHT

def simple_contrast(map_h_ori, map_h_pos, map_h_neg, mask, cfg):
    map_h_ori = map_h_ori * mask
    map_h_pos = torch.sigmoid(map_h_pos) * mask
    map_h_neg = torch.sigmoid(map_h_neg) * mask

    batch_size = map_h_ori.shape[0]
    map_h_ori_reshape = map_h_ori.reshape(batch_size, -1)
    map_h_pos_reshape = map_h_pos.reshape(batch_size, -1)
    map_h_neg_reshape = map_h_neg.reshape(batch_size, -1)
    pos = torch.cosine_similarity(map_h_ori_reshape, map_h_pos_reshape, dim=1)
    neg = torch.cosine_similarity(map_h_ori_reshape, map_h_neg_reshape, dim=1)
    logit = torch.stack((pos, neg), 1) # [b, 2]
    contras_loss = -F.log_softmax(logit, 1)
    loss_value = contras_loss[:, 0].mean()
    return loss_value * cfg.WEIGHT

def triplet_contrast(map_h_ori, map_h_pos, map_h_neg, mask, cfg):
    map_h_ori = torch.sigmoid(map_h_ori) * mask
    map_h_pos = torch.sigmoid(map_h_pos) * mask
    map_h_neg = torch.sigmoid(map_h_neg) * mask

    batch_size = map_h_ori.shape[0]
    map_h_ori_reshape = map_h_ori.reshape(batch_size, -1)
    map_h_pos_reshape = map_h_pos.reshape(batch_size, -1)
    map_h_neg_reshape = map_h_neg.reshape(batch_size, -1)
    contras_loss = F.triplet_margin_loss(map_h_ori_reshape, map_h_pos_reshape, map_h_neg_reshape, margin=0.1)
    return contras_loss * cfg.WEIGHT