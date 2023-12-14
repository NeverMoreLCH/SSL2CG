import os
import json
import tqdm
import torch
import torchtext
import torch.utils.data
import torch.nn.functional as F
from .utils import iou


class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(DatasetBase, self).__init__()

        assert cfg.SLIDING_WINDOW == True
        assert cfg.TXT_INPUT_TYPE == 'glove'
        assert isinstance(cfg.OUTPUT_NUM_CLIPS, list) == True
        assert isinstance(cfg.NUM_ANCHORS, list) == True

        self.cfg = cfg
        self.split = split
        self.scale_duration = cfg.TIME_UNIT * cfg.INPUT_NUM_CLIPS
    
    def get_annotations(self):
        annotations = []
        with open(os.path.join(self.cfg.DATA_DIR, f'{self.split}.json'), 'r') as f:
            info = json.load(f)
        
        for vid, anns in info.items():
            duration = anns['duration']
            for timestamp, sent in zip(anns['timestamps'], anns['sentences']):
                annotations.append({
                    'video': vid, 'duration': duration,
                    'times': timestamp, 'sent': sent})
        return annotations
    
    def get_sent_feats(self):
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        embd = torchtext.vocab.GloVe(name="6B", dim=300)
        embd.itos.extend(['<unk>'])
        embd.stoi['<unk>'] = embd.vectors.shape[0]
        embd.vectors = torch.cat([embd.vectors, torch.zeros(1, 300)], dim=0)

        sent_feats, sent_masks = [], []
        for annotation in tqdm.tqdm(self.annotations, desc='Loading sent feats'):
            sent = annotation['sent']
            token = tokenizer(sent)
            feat = embd.get_vecs_by_tokens(token, lower_case_backup=True)
            sent_feats.append(feat)
            sent_masks.append(torch.ones(feat.shape[0], 1))
        return sent_feats, sent_masks

    def get_gt_map(self, momnet, num_clips):
        gt_maps, masks = [], []
        for num_out_clips, num_anchors in zip(self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS):
            downsample_rate = self.cfg.INPUT_NUM_CLIPS // num_out_clips
            scale_num_clips = max(num_clips // downsample_rate, 1)
            scale_duration = self.cfg.TIME_UNIT * downsample_rate * scale_num_clips
            
            iou2d = torch.ones(scale_num_clips, num_anchors)
            mask = torch.zeros(max(scale_num_clips, num_out_clips), num_anchors, dtype=torch.bool)
            for i in range(1, min(num_anchors, scale_num_clips) + 1):
                iou2d[scale_num_clips - i, i:] = 0
            
            iou2d_moment = iou2d.nonzero().float()
            iou2d_idx = iou2d.nonzero(as_tuple=True)

            iou2d_moment[:, 1] += (iou2d_moment[:, 0] + 1)
            iou2d_moment[:, 0] *= (scale_duration / scale_num_clips)
            iou2d_moment[:, 1] *= (scale_duration / scale_num_clips)
            iou_ = iou(iou2d_moment, momnet)
            
            mask[iou2d_idx] = 1
            iou2d[iou2d_idx] = iou_
            if num_clips < self.cfg.INPUT_NUM_CLIPS:
                iou2d = F.pad(iou2d, [0, 0, 0, num_out_clips - scale_num_clips])
            
            gt_maps.append(iou2d)
            masks.append(mask)
        return gt_maps, masks

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.annotations)
