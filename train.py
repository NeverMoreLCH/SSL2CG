import os
import time
import tqdm
import torch
import argparse
import models.loss as loss
from config import cfg, update_config
from models.TAN import MultiScale_TAN
from torch.utils.data import DataLoader
from datasets.charades import CharadesSampleCG
from torch.utils.tensorboard import SummaryWriter
from datasets import train_collate_fn, train_sample_collate_fn
from datasets.utils import recover_to_single_map, nms_acc, miou


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--seed', help='seed', default=0, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--mode', default='train', help='run test epoch only')
    parser.add_argument('--split', help='test split', type=str)
    parser.add_argument('--no_save', default=False, action="store_true", help='don\'t save checkpoint')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers is not None:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATASET.DATA_DIR = os.path.join(args.dataDir, config.DATASET.DATA_DIR)
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.tag:
        config.TAG = args.tag

@torch.no_grad()
def train_log(predict_maps, gt_maps, video_masks):
    predict_maps = [torch.sigmoid(predict_map) * video_mask 
                        for predict_map, video_mask in zip(predict_maps, video_masks)]
    predict_maps = recover_to_single_map(predict_maps)
    gt_maps = recover_to_single_map(gt_maps)

    rank1_5, rank5_5 = 0, 0
    for predict_map, gt_map in zip(predict_maps, gt_maps):
        rank1_5 += nms_acc(predict_map, gt_map, 1, 0.5, cfg.TEST.NMS_THRESH)
        rank5_5 += nms_acc(predict_map, gt_map, 5, 0.5, cfg.TEST.NMS_THRESH)
    rank1_5 /= predict_maps.shape[0]
    rank5_5 /= predict_maps.shape[0]

    log = f"Epoch={epoch:03}, Iteration={iteration:06}, Loss={ori_loss.detach().item():6.5f}, " \
          f"Rank1@0.5={rank1_5:05.3%}, Rank5@0.5={rank5_5:05.3%}, " \
          f"Time={time.time() - time_:2.4f}"
    print(log)
    with open(os.path.join(cfg.LOG_DIR, 'train.log'), 'a') as f:
        f.write(log + '\n')
    writer.add_scalar('Train BCELoss', ori_loss.detach().item(), global_step=iteration)
    writer.add_scalar('Train Rank1@0.5', rank1_5, global_step=iteration)
    writer.add_scalar('Train Rank5@0.5', rank5_5, global_step=iteration)

@torch.no_grad()
def eval_log(eval_acc, eval_miou, desc):
    eval_acc = eval_acc.flatten().tolist()
    eval_miou = eval_miou.item()
    log = "Rank1@0.5={0:05.3%}, Rank1@0.7={1:05.3%}, mIoU={4:05.3%}" \
          "\nRank5@0.5={2:05.3%}, Rank5@0.7={3:05.3%}\n".format(*eval_acc, eval_miou)
    print(log)
    with open(os.path.join(cfg.LOG_DIR, 'train.log'), 'a') as f:
        f.write(log + '\n')
    writer.add_scalar(f'{desc} Rank1@0.5', eval_acc[0], global_step=iteration)
    writer.add_scalar(f'{desc} Rank1@0.7', eval_acc[1], global_step=iteration)
    writer.add_scalar(f'{desc} Rank5@0.5', eval_acc[2], global_step=iteration)
    writer.add_scalar(f'{desc} Rank5@0.7', eval_acc[3], global_step=iteration)
    writer.add_scalar(f'{desc} mIoU', eval_miou, global_step=iteration)
    writer.add_scalar(f'{desc} Metric', eval_acc[0] + eval_acc[1] + eval_miou, global_step=iteration)
    return eval_acc[0] + eval_acc[1] + eval_miou

@torch.no_grad()
def eval_model(model, test_loader, desc):
    eval_acc = torch.zeros((2,2))
    eval_miou = torch.zeros((1, ))
    total_num = 0
    for sent_feats, sent_masks, video_feats, video_masks, gt_maps in tqdm.tqdm(test_loader, desc=desc):
        sent_feats, sent_masks, video_feats = sent_feats.cuda(), \
                                                    sent_masks.cuda(), video_feats.cuda()
        gt_maps = [gt_map.cuda() for gt_map in gt_maps]
        video_masks = [video_mask.cuda() for video_mask in video_masks]
        predict_maps = model(sent_feats, sent_masks, video_feats, video_masks)

        predict_maps = [torch.sigmoid(predict_map) * video_mask 
                            for predict_map, video_mask in zip(predict_maps, video_masks)]
        predict_maps = recover_to_single_map(predict_maps)
        gt_maps = recover_to_single_map(gt_maps)
        for predict_map, gt_map in zip(predict_maps, gt_maps):
            eval_acc[0, 0] += nms_acc(predict_map, gt_map, 1, 0.5, cfg.TEST.NMS_THRESH)
            eval_acc[0, 1] += nms_acc(predict_map, gt_map, 1, 0.7, cfg.TEST.NMS_THRESH)
            eval_acc[1, 0] += nms_acc(predict_map, gt_map, 5, 0.5, cfg.TEST.NMS_THRESH)
            eval_acc[1, 1] += nms_acc(predict_map, gt_map, 5, 0.7, cfg.TEST.NMS_THRESH)
            eval_miou += miou(predict_map, gt_map)
            total_num += 1
    eval_score = eval_log(eval_acc / total_num, eval_miou / total_num, desc)
    return eval_score

def Loss(predict_maps, gt_maps, masks, cfg, weight=None):
    loss_sum = 0
    for predict_map, gt_map, mask in zip(predict_maps, gt_maps, masks):
        if cfg.NAME == 'bce_rescale_loss':
            loss_sum += getattr(loss, cfg.NAME)(predict_map, mask, gt_map, cfg.PARAMS)
        else:
            loss_sum += getattr(loss, cfg.NAME)(predict_map, mask, gt_map, cfg.PARAMS, batch_weight=weight)
    return loss_sum

if __name__ == '__main__':
    torch.manual_seed(1111)
    torch.cuda.manual_seed(11111)
    args = parse_args()
    reset_config(cfg, args)
    if os.path.exists(os.path.join(cfg.LOG_DIR, 'train.log')) or \
            os.path.exists(os.path.join(cfg.MODEL_DIR, 'MS-2D-TAN.pt')):
        raise RuntimeError('Log file or checkpoints file exists.')
    with open(os.path.join(cfg.LOG_DIR, 'train.log'), 'a') as f:
        f.write(cfg.__repr__() + '\n')
    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    writer = SummaryWriter(cfg.LOG_DIR)
    
    train_data = CharadesSampleCG(cfg.DATASET, 'train')
    tt_data = CharadesSampleCG(cfg.DATASET, 'test_trivial')
    nw_data = CharadesSampleCG(cfg.DATASET, 'novel_word')
    nc_data = CharadesSampleCG(cfg.DATASET, 'novel_composition')

    train_loader = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, 
                            shuffle=cfg.TRAIN.SHUFFLE, num_workers=cfg.WORKERS, 
                            pin_memory=True, collate_fn=train_sample_collate_fn)
    tt_loader = DataLoader(tt_data, batch_size=cfg.TEST.BATCH_SIZE, 
                            shuffle=False, num_workers=cfg.WORKERS, 
                            pin_memory=True, collate_fn=train_collate_fn)
    nw_loader = DataLoader(nw_data, batch_size=cfg.TEST.BATCH_SIZE, 
                            shuffle=False, num_workers=cfg.WORKERS, 
                            pin_memory=True, collate_fn=train_collate_fn)
    nc_loader = DataLoader(nc_data, batch_size=cfg.TEST.BATCH_SIZE, 
                            shuffle=False, num_workers=cfg.WORKERS, 
                            pin_memory=True, collate_fn=train_collate_fn)
    
    model = MultiScale_TAN(cfg.MODEL).cuda()

    assert cfg.OPTIM.NAME == 'Adam'
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)

    old_acc = 0
    iteration, epoch, time_ = 0, 0, time.time()
    while epoch <= cfg.TRAIN.MAX_EPOCH:
        epoch += 1
        model.train()
        for ori_feat, neg_feat, pos_feat, sent_masks, video_masks, gt_maps in train_loader:
            iteration += 1
            ori_feat = [feat.cuda(non_blocking=True) for feat in ori_feat]
            neg_feat = [feat.cuda(non_blocking=True) for feat in neg_feat]
            pos_feat = [feat.cuda(non_blocking=True) for feat in pos_feat]
            sent_masks = sent_masks.cuda(non_blocking=True)
            gt_maps = [gt_map.cuda(non_blocking=True) for gt_map in gt_maps]
            video_masks = [video_mask.cuda(non_blocking=True) for video_mask in video_masks]

            ori_predict_maps = model(ori_feat[0], sent_masks, ori_feat[1], video_masks)
            ori_loss = Loss(ori_predict_maps, gt_maps, video_masks, cfg.LOSS.ORI)
            ori_loss.backward()
            [param.grad.div_(2) for param in model.clip_module.parameters()]
            [param.grad.div_(2) for param in model.lang_module.parameters()]
            optimizer.step()
            optimizer.zero_grad()

            neg_predict_maps = model(neg_feat[0], sent_masks, neg_feat[1], video_masks)
            pos_predict_maps = model(pos_feat[0], sent_masks, pos_feat[1], video_masks)
            neg_loss = Loss(neg_predict_maps, gt_maps, video_masks, cfg.LOSS.NEG, neg_feat[2])
            pos_loss = Loss(pos_predict_maps, gt_maps, video_masks, cfg.LOSS.POS, pos_feat[2])
            if cfg.LOSS.CONTRAST.NAME:
                contrast_loss = 0
                for gt_p, neg_p, pos_p, v_mask in zip(gt_maps, neg_predict_maps, pos_predict_maps, video_masks):
                    contrast_loss += getattr(loss, cfg.LOSS.CONTRAST.NAME)(gt_p, pos_p, neg_p, v_mask, cfg.LOSS.CONTRAST.PARAMS)
            sample_loss = neg_loss + pos_loss + contrast_loss
            sample_loss.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.25)
            [param.grad.div_(2) for param in model.clip_module.parameters()]
            [param.grad.div_(2) for param in model.lang_module.parameters()]
            optimizer.step()
            optimizer.zero_grad()

            if iteration % 200 == 0:
                writer.add_scalar('Ori Loss', ori_loss.detach().cpu().item(), global_step=iteration)
                writer.add_scalar('Neg Loss', neg_loss.detach().cpu().item(), global_step=iteration)
                writer.add_scalar('Pos Loss', pos_loss.detach().cpu().item(), global_step=iteration)
                writer.add_scalar('Contrast Loss', contrast_loss.detach().cpu().item(), global_step=iteration)
                train_log(ori_predict_maps, gt_maps, video_masks)
                time_ = time.time()

        model.eval()
        new_acc = .0
        new_acc += eval_model(model, tt_loader, 'TT')
        new_acc += eval_model(model, nw_loader, 'NW')
        new_acc += eval_model(model, nc_loader, 'NC')

        if new_acc > old_acc:
            old_acc = new_acc
            torch.save(model.state_dict(), 
                        os.path.join(cfg.MODEL_DIR, 'MS-2D-TAN.pt'))
