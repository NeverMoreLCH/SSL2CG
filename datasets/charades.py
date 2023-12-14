import os
import csv
import h5py
import tqdm
import nltk
import json
import torch
import random
import torchtext
import torch.utils.data
from .utils import pre_caption
import torch.nn.functional as F
from .dataset import DatasetBase

random.seed(1111)

class CharadesDataset(DatasetBase):
    def __init__(self, cfg, split):
        super(CharadesDataset, self).__init__(cfg, split)

        # Load Annotations
        self.annotations = self.get_annotations()
        if split != 'train':
            self.annotations = sorted(self.annotations, key=lambda ann: ann['duration'])
        # Tokenizer&Word Embedding
        self.sent_feats, self.sent_masks = self.get_sent_feats()
        # Video features
        self.video_feats = self.get_video_feats()
        # Ground truth map and mask
        self.gt_maps, self.video_masks = self.get_gt_maps()
    
    def get_annotations(self):
        durations = {}
        with open(os.path.join(self.cfg.DATA_DIR, f'Charades_v1_{self.split}.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                durations[row['id']] = float(row['length'])

        with open(os.path.join(self.cfg.DATA_DIR, f'charades_sta_{self.split}.txt'), 'r') as f:
            annotations = []
            for line in f:
                anno, sent = line.split("##")
                sent = sent.split('.\n')[0]
                vid, s_time, e_time = anno.split(" ")
                duration = durations[vid]
                s_time = float(s_time)
                e_time = min(float(e_time), duration)
                if s_time < e_time:
                    annotations.append(
                        {'video': vid, 'times': [s_time, e_time], 
                        'duration': duration, 'sent': sent})
        return annotations
    
    def get_video_feats(self):
        vids = set()
        for annotation in self.annotations:
            vid = annotation['video']
            vids.add(vid)
        
        feats = {}
        with h5py.File(os.path.join(self.cfg.DATA_DIR, f'{self.cfg.VIS_INPUT_TYPE}.hdf5'), 'r') as f:
            for vid in tqdm.tqdm(vids, desc='Loading video feats'):
                feat = torch.from_numpy(f[vid][:]).float()
                if self.cfg.NORMALIZE:
                    feat = F.normalize(feat,dim=1)
                feats[vid] = feat.t()
        return feats

    def get_gt_maps(self):
        if self.split == 'train':
            return None, None
        
        gt_maps, video_masks = [], []
        for annotation in tqdm.tqdm(self.annotations, desc='Loading gt maps'):
            vid = annotation['video']
            num_clips = self.video_feats[vid].shape[1]
            gt_s_time, gt_e_time = annotation['times']
            gt_map, video_mask = self.get_gt_map(torch.tensor([gt_s_time, gt_e_time]), self.cfg.INPUT_NUM_CLIPS)
            gt_maps.append(gt_map)
            video_masks.append(video_mask)
        
        for vid in self.video_feats.keys():
            video_feat = self.video_feats[vid]
            if video_feat.shape[1] > self.cfg.INPUT_NUM_CLIPS:
                video_feat = video_feat[:, :self.cfg.INPUT_NUM_CLIPS]
            self.video_feats[vid] = video_feat

        return gt_maps, video_masks

    def __getitem__(self, index):
        vid = self.annotations[index]['video']
        video_feat = self.video_feats[vid]
        num_clips = video_feat.shape[1]
        
        if num_clips < self.cfg.INPUT_NUM_CLIPS:
            video_feat = F.pad(video_feat, [0, self.cfg.INPUT_NUM_CLIPS - num_clips, 0, 0])

        if self.split != 'train':
            return self.sent_feats[index], self.sent_masks[index], \
                        video_feat, self.video_masks[index], self.gt_maps[index]
        else:
            gt_s_time, gt_e_time = self.annotations[index]['times']
            
            if num_clips > self.cfg.INPUT_NUM_CLIPS:
                s_idx = random.randrange(num_clips - self.cfg.INPUT_NUM_CLIPS)
                e_idx = s_idx + self.cfg.INPUT_NUM_CLIPS
                video_feat = video_feat[:, s_idx:e_idx]
                gt_s_time = gt_s_time - s_idx * self.cfg.TIME_UNIT
                gt_e_time = gt_e_time - s_idx * self.cfg.TIME_UNIT
            
            gt_map, video_mask = self.get_gt_map(torch.tensor([gt_s_time, gt_e_time]), 
                                            self.cfg.INPUT_NUM_CLIPS)
            return self.sent_feats[index], self.sent_masks[index], \
                        video_feat, video_mask, gt_map


class CharadesCGDataset(CharadesDataset):
    def get_annotations(self):
        return DatasetBase.get_annotations(self)
    
    def get_video_feats(self):
        _tmp = self.cfg.DATA_DIR
        self.cfg.DATA_DIR = 'data/Charades'
        video_feats = super(CharadesCGDataset, self).get_video_feats()
        self.cfg.DATA_DIR = _tmp
        return video_feats


class CharadesSample(CharadesDataset):
    def __init__(self, cfg, split):
        super(CharadesSample, self).__init__(cfg, split)

        self.tag2bit_mask = {'N': 0b00001, 'V':0b00010, 'J': 0b00100, 'R':0b01000, 'O':0b10000}
        self.tag2weight = {'N': 1, 'V': 1, 'J': 0.6, 'R': 0.6, 'O': 0}
 
        # Part of speech tag
        self.sent_tags, self.neg_tag_mask, self.pos_tag_mask = self.get_sent_tags()
        # Similarity between video feat and language
        self.video_sims = self.get_video_sims()
        # Normalized video length to self.cfg.INPUT_NUM_CLIPS
        for vid in self.video_feats.keys():
            video_feat = self.video_feats[vid]
            num_clips = video_feat.shape[1]
            if num_clips < self.cfg.INPUT_NUM_CLIPS:
                video_feat = F.pad(video_feat, [0, self.cfg.INPUT_NUM_CLIPS - num_clips, 0, 0])
            self.video_feats[vid] = video_feat
    
    def get_sent_tags(self):
        if self.split != 'train':
            return None, None, None
        
        neg_tag_mask = 0b00000000
        for tag_mask in self.cfg.SAMPLE.NEG.Q_PARAMS.MASK_WORD_TYPE.split('_', 1)[1]:
            if tag_mask in ['N', 'V', 'J', 'R', 'O']:
                neg_tag_mask |= self.tag2bit_mask[tag_mask]
            else:
                raise NotImplementedError
        pos_tag_mask = 0b00000000
        for tag_mask in self.cfg.SAMPLE.POS.Q_PARAMS.MASK_WORD_TYPE.split('_', 1)[1]:
            if tag_mask in ['N', 'V', 'J', 'R', 'O']:
                pos_tag_mask |= self.tag2bit_mask[tag_mask]
            else:
                raise NotImplementedError


        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        sent_tags = []
        for annotation in tqdm.tqdm(self.annotations, desc='Loading sent tags'):
            sent = annotation['sent']
            token = tokenizer(sent)
            sent_tag, sent_weight = [], []
            for tag in nltk.pos_tag(token):
                if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    sent_tag.append(self.tag2bit_mask['N'])
                    sent_weight.append(self.tag2weight['N'])
                elif tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    sent_tag.append(self.tag2bit_mask['V'])
                    sent_weight.append(self.tag2weight['V'])
                elif tag[1] in ['JJ', 'JJR', 'JJS']:
                    sent_tag.append(self.tag2bit_mask['J'])
                    sent_weight.append(self.tag2weight['J'])
                elif tag[1] in ['RB', 'RBR', 'RBS', 'RP']:
                    sent_tag.append(self.tag2bit_mask['R'])
                    sent_weight.append(self.tag2weight['R'])
                else:
                    sent_tag.append(self.tag2bit_mask['O'])
                    sent_weight.append(self.tag2weight['O'])
            sent_tags.append((torch.tensor(sent_tag, dtype=torch.int32),
                                torch.tensor(sent_weight)))
        return sent_tags, neg_tag_mask, pos_tag_mask
    
    def get_video_sims(self):
        if self.split != 'train':
            return None
        
        video_sims = []
        with open(os.path.join(self.cfg.DATA_DIR, 'frame_query_sim_sec0.64_TCL_all.json'), 'r') as f:
            sim_data = json.load(f)
        for annotation in tqdm.tqdm(self.annotations, desc='Loading video sims'):
                vid = annotation['video']
                sent = annotation['sent']
                sent = pre_caption(sent, 30).replace(' ', '_')
                video_sim = sim_data[vid + '_' + sent]
                video_sim = torch.tensor(video_sim[:self.video_feats[vid].shape[1]])
                video_sim = (video_sim - video_sim.min()) / (video_sim.max() - video_sim.min())
                video_sim[video_sim == 0] += 1e-9
                if video_sim.shape[0] < self.cfg.INPUT_NUM_CLIPS:
                    video_sim = F.pad(video_sim, [0, self.cfg.INPUT_NUM_CLIPS - video_sim.shape[0]])
                video_sims.append(video_sim)
        return video_sims

    def get_sent_sample(self, sent_tag, tag_mask, cfg):
        """
        
        : param sent_tag: Tag and weight of sentence, tuple(list[L], list[L])
        : param tag_mask: Bit tag mask, int
        : return: (sample mask, weight), where 0 reprent to mask, tuple(list[L], int)
        """
        op1 = cfg.Q_PARAMS.MASK_WORD_TYPE[0:4]
        op2 = cfg.Q_PARAMS.MASK_WORD_TYPE[4]

        sent_tag, sent_weight = sent_tag
        sample_mask = (sent_tag & tag_mask) != 0
        if op1 == 'Rand':
            tmp_mask = torch.rand_like(sample_mask, dtype=torch.float32) < cfg.Q_PARAMS.MASK_RATE
            sample_idx = (sample_mask * tmp_mask).nonzero()
        elif op1 == 'Step':
            sample_idx = sample_mask.nonzero()[:, ::cfg.Q_PARAMS.MASK_STEP]
        elif op1 == 'Firs':
            assert cfg.Q_PARAMS.MASK_WORD_TYPE[0:5] == 'First'
            op2 = cfg.Q_PARAMS.MASK_WORD_TYPE[5]
            sample_idx = sample_mask.nonzero()[:, :cfg.Q_PARAMS.MASK_NUM]
        elif op1 == 'Last':
            sample_idx = sample_mask.nonzero()[:, -cfg.Q_PARAMS.MASK_NUM:]
        else:
            raise NotImplementedError
        sample_mask = torch.ones_like(sample_mask)
        sample_mask[sample_idx] = False

        batch_weight = 1
        if cfg.BATCH_WEIGHT:
            sample_weight = sent_weight[~sample_mask].sum()
            total_weight = sent_weight.sum()
            batch_weight = sample_weight / total_weight
        
        assert op2 == 'M'
        return sample_mask, batch_weight

    def get_video_sample(self, video_sim, gt_times, cfg):
        tmp_sim = video_sim.clone()
        gt_s_idx = max(min(round(gt_times[0] / self.cfg.TIME_UNIT), video_sim.shape[0]), 0)
        gt_e_idx = max(min(round(gt_times[1] / self.cfg.TIME_UNIT), video_sim.shape[0]), 0)
        gt_idx = torch.arange(gt_s_idx, gt_e_idx)
        tmp_sim[gt_idx] = 0
        ngt_sort, ngt_idx = torch.sort(tmp_sim, descending=True)
        ngt_idx = ngt_idx[ngt_sort != 0]

        sample_mask = torch.ones_like(video_sim, dtype=torch.bool)
        if cfg.V_PARAMS.get('IN_GT_MASK_RATE'):
            tmp_mask = torch.rand_like(gt_idx, dtype=torch.float32) < cfg.V_PARAMS.IN_GT_MASK_RATE
            gt_mask_idx = gt_idx[tmp_mask]
            sample_mask[gt_mask_idx] = False
        ngt_mask_idx = ngt_idx[:round(ngt_idx.shape[0] * cfg.V_PARAMS.OUT_GT_RATE)]
        tmp_mask = torch.rand_like(ngt_mask_idx, dtype=torch.float32) < cfg.V_PARAMS.OUT_GT_MASK_RATE
        ngt_mask_idx = ngt_mask_idx[tmp_mask]
        sample_mask[ngt_mask_idx] = False

        batch_weight = 1
        if cfg.BATCH_WEIGHT:
            sample_weight = video_sim[~sample_mask].sum()
            total_weight = video_sim.sum()
            batch_weight = sample_weight / total_weight

        return sample_mask, batch_weight
    
    def __getitem__(self, index):
        vid = self.annotations[index]['video']
        if self.split != 'train':
            return self.sent_feats[index], self.sent_masks[index], \
                    self.video_feats[vid], self.video_masks[index], \
                    self.gt_maps[index]
        
        sent_feat = self.sent_feats[index]
        video_feat = self.video_feats[vid]
        sent_tag = self.sent_tags[index]
        video_sim = self.video_sims[index]
        
        num_clips = video_feat.shape[1]
        gt_s_time, gt_e_time = self.annotations[index]['times']
        if num_clips > self.cfg.INPUT_NUM_CLIPS:
            s_idx = random.randrange(num_clips - self.cfg.INPUT_NUM_CLIPS)
            e_idx = s_idx + self.cfg.INPUT_NUM_CLIPS
            video_feat = video_feat[:, s_idx:e_idx]
            video_sim = video_sim[s_idx:e_idx]
            gt_s_time = gt_s_time - s_idx * self.cfg.TIME_UNIT
            gt_e_time = gt_e_time - s_idx * self.cfg.TIME_UNIT
        
        gt_map, video_mask = self.get_gt_map(torch.tensor([gt_s_time, gt_e_time]), 
                                        self.cfg.INPUT_NUM_CLIPS)
        
        ori_sent_feat = sent_feat
        ori_video_feat = video_feat
        ori_feat = (ori_sent_feat, ori_video_feat)
        
        s_flag, v_flag = False, False
        if self.cfg.SAMPLE.NEG.MASK_MODE in ['Q', 'QandV']:
            s_flag = True
        if self.cfg.SAMPLE.NEG.MASK_MODE in ['V', 'QandV']:
            v_flag = True
        if self.cfg.SAMPLE.NEG.MASK_MODE == 'QorV':
            s_flag = random.random() < 0.5
            v_flag = not s_flag

        if s_flag:
            neg_sent_mask, neg_sent_weight = self.get_sent_sample(sent_tag, self.neg_tag_mask, 
                                                                            self.cfg.SAMPLE.NEG)
            neg_sent_feat = ori_sent_feat * neg_sent_mask[..., None]
        else:
            neg_sent_feat = ori_sent_feat
            neg_sent_weight = 0
        if v_flag:
            neg_video_mask, neg_video_weight = self.get_video_sample(video_sim, 
                                                    [gt_s_time, gt_e_time], self.cfg.SAMPLE.NEG)
            neg_video_feat = ori_video_feat * neg_video_mask
        else:
            neg_video_feat = ori_video_feat
            neg_video_weight = 0
        neg_feat = (neg_sent_feat, neg_video_feat, neg_sent_weight + neg_video_weight)
        
        s_flag, v_flag = False, False
        if self.cfg.SAMPLE.POS.MASK_MODE in ['Q', 'QandV']:
            s_flag = True
        if self.cfg.SAMPLE.POS.MASK_MODE in ['V', 'QandV']:
            v_flag = True
        if self.cfg.SAMPLE.POS.MASK_MODE == 'QorV':
            s_flag = random.random() < 0.5
            v_flag = not s_flag
        
        if s_flag:
            pos_sent_mask, pos_sent_weight = self.get_sent_sample(sent_tag, self.pos_tag_mask, 
                                                                            self.cfg.SAMPLE.POS)
            pos_sent_feat = ori_sent_feat * pos_sent_mask[..., None]
        else:
            pos_sent_feat = ori_sent_feat
            pos_sent_weight = 0
        if v_flag:
            pos_video_mask, pos_video_weight = self.get_video_sample(video_sim, 
                                                    [gt_s_time, gt_e_time], self.cfg.SAMPLE.POS)
            pos_video_feat = ori_video_feat * pos_video_mask
        else:
            pos_video_feat = ori_video_feat
            pos_video_weight = 0
        pos_feat = (pos_sent_feat, pos_video_feat, pos_sent_weight + pos_video_weight)

        return ori_feat, neg_feat, pos_feat, self.sent_masks[index], video_mask, gt_map
        

class CharadesSampleCG(CharadesSample):
    def get_annotations(self):
        return DatasetBase.get_annotations(self)
    
    def get_video_feats(self):
        _tmp = self.cfg.DATA_DIR
        self.cfg.DATA_DIR = 'data/Charades'
        video_feats = super(CharadesSampleCG, self).get_video_feats()
        self.cfg.DATA_DIR = _tmp
        return video_feats
