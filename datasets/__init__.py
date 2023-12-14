from .charades import *

def train_collate_fn(batch):
    sent_feats, sent_masks, video_feats, video_masks, gt_maps = [], [], [], [], []
    for sent_feat, sent_mask, video_feat, video_mask, gt_map in batch:
        sent_feats.append(sent_feat)
        sent_masks.append(sent_mask)
        video_feats.append(video_feat)
        video_masks.append(video_mask)
        gt_maps.append(gt_map)
    
    sent_feats = torch.nn.utils.rnn.pad_sequence(sent_feats, batch_first=True, padding_value=0)
    sent_masks = torch.nn.utils.rnn.pad_sequence(sent_masks, batch_first=True, padding_value=0)
    video_feats = torch.stack(video_feats)
    video_masks = [torch.stack(video_mask) for video_mask in zip(*video_masks)]
    gt_maps = [torch.stack(gt_map) for gt_map in zip(*gt_maps)]
    return sent_feats, sent_masks, video_feats, video_masks, gt_maps

def eval_collate_fn(batch):
    sent_feats, sent_masks, video_feats, video_masks, gt_maps = [], [], [], [], []
    for sent_feat, sent_mask, video_feat, video_mask, gt_map in batch:
        sent_feats.append(sent_feat)
        sent_masks.append(sent_mask)
        video_feats.append(video_feat.t())
        video_masks.append(video_mask)
        gt_maps.append(gt_map)
    
    sent_feats = torch.nn.utils.rnn.pad_sequence(sent_feats, batch_first=True, padding_value=0)
    sent_masks = torch.nn.utils.rnn.pad_sequence(sent_masks, batch_first=True, padding_value=0)
    video_feats = torch.nn.utils.rnn.pad_sequence(video_feats, batch_first=True, padding_value=0).transpose(1, 2)
    video_masks = [torch.nn.utils.rnn.pad_sequence(video_mask, batch_first=True, padding_value=0) for video_mask in zip(*video_masks)]
    gt_maps = [torch.nn.utils.rnn.pad_sequence(gt_map, batch_first=True, padding_value=0) for gt_map in zip(*gt_maps)]
    return sent_feats, sent_masks, video_feats, video_masks, gt_maps

def train_sample_collate_fn(batch):
    ori_sent_feats, ori_video_feats = [], []
    neg_sent_feats, neg_video_feats, neg_weight = [], [], []
    pos_sent_feats, pos_video_feats, pos_weight = [], [], []
    sent_masks, video_masks, gt_maps = [], [], []
    for ori_feat, neg_feat, pos_feat, sent_mask, video_mask, gt_map in batch:
        ori_sent_feats.append(ori_feat[0])
        ori_video_feats.append(ori_feat[1])
        neg_sent_feats.append(neg_feat[0])
        neg_video_feats.append(neg_feat[1])
        neg_weight.append(neg_feat[2])
        pos_sent_feats.append(pos_feat[0])
        pos_video_feats.append(pos_feat[1])
        pos_weight.append(pos_feat[2])
        sent_masks.append(sent_mask)
        video_masks.append(video_mask)
        gt_maps.append(gt_map)
    
    ori_sent_feats = torch.nn.utils.rnn.pad_sequence(ori_sent_feats, batch_first=True, padding_value=0)
    neg_sent_feats = torch.nn.utils.rnn.pad_sequence(neg_sent_feats, batch_first=True, padding_value=0)
    pos_sent_feats = torch.nn.utils.rnn.pad_sequence(pos_sent_feats, batch_first=True, padding_value=0)
    ori_video_feats = torch.stack(ori_video_feats)
    neg_video_feats = torch.stack(neg_video_feats)
    pos_video_feats = torch.stack(pos_video_feats)
    neg_weight = torch.tensor(neg_weight)
    pos_weight = torch.tensor(pos_weight)
    sent_masks = torch.nn.utils.rnn.pad_sequence(sent_masks, batch_first=True, padding_value=0)
    video_masks = [torch.stack(video_mask) for video_mask in zip(*video_masks)]
    gt_maps = [torch.stack(gt_map) for gt_map in zip(*gt_maps)]
    return (ori_sent_feats, ori_video_feats), \
            (neg_sent_feats, neg_video_feats, neg_weight), \
            (pos_sent_feats, pos_video_feats, pos_weight), \
                sent_masks, video_masks, gt_maps
