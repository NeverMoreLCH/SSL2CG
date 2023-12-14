import re
import torch

def iou(iou2d_moment, moment):
    """
    
    :param iou2d_moment: Moment of iou2d, shape=[N * N, 2]
    :param moment: Moment, Tensor[s, e]
    :return : Iou of each pos in iou2d, shape=[N * N, 1]
    """
    inter = iou2d_moment[:, 1].fmin(moment[1]) - iou2d_moment[:, 0].fmax(moment[0])
    union = iou2d_moment[:, 1].fmax(moment[1]) - iou2d_moment[:, 0].fmin(moment[0])
    return inter.clamp(min=0) / union

def avgfeats(feat, num_pre_clips):
    """
    
    :param feat: Raw video features, shape=[?, D]
    :param num_pre_clips: Num of clips for each video feature, Int
    :return : Feature with fixed shape, shape=[num_pre_clips, D]
    """
    num_src_clips, D = feat.shape
    idxs = torch.arange(0, num_pre_clips + 1, 1.) * num_src_clips / num_pre_clips
    idxs = idxs.round().long().clamp(min=0, max=num_src_clips - 1)
    feat_ = torch.empty(num_pre_clips, D)
    for i in range(num_pre_clips):
        if idxs[i] < idxs[i + 1]:
            feat_[i] = feat[idxs[i]:idxs[i + 1]].mean(dim=0)
        else:
            feat_[i] = feat[idxs[i]]
    return feat_

def recover_to_single_map(joint_probs):
    batch_size, map_size, _ = joint_probs[0].shape
    score_map = joint_probs[0].new_zeros(batch_size, map_size, map_size)
    for prob in joint_probs:
        scale_num_clips, scale_num_anchors = prob.shape[1:]
        dilation = map_size//scale_num_clips
        for i in range(scale_num_anchors):
            score_map[...,:map_size//dilation*dilation:dilation,(i+1)*dilation-1] = torch.max(
                score_map[...,:map_size//dilation*dilation:dilation,(i+1)*dilation-1].clone(), prob[...,i])
    return score_map

def nms_acc(predict_map, gt_map, n, m, thresh):
    assert predict_map.shape == gt_map.shape
    
    N = predict_map.shape[-1]
    predict_map = predict_map.reshape(-1, )
    gt_map = gt_map.reshape(-1, )
    scores, idxs = torch.sort(predict_map, descending=True)
    gt_map = gt_map[idxs]
    idxs = torch.stack((idxs // N, idxs % N), dim=1)
    idxs[:, 1] += (idxs[:, 0] + 1)
    
    scores.zero_()
    count = 0
    for i in range(scores.shape[0]):
        if scores[i] == 1:
            continue
        if gt_map[i] >= m:
            return 1
        count += 1
        if count == n:
            return 0
        mask = iou(idxs[i + 1:], idxs[i]) > thresh
        scores[i + 1:][mask] = 1
    return 0

def miou(predict_map, gt_map):
    idx = predict_map.argmax()
    return gt_map.flatten()[idx].cpu()

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption