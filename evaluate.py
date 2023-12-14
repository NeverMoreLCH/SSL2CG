import os
import tqdm
import torch
import argparse
from config import cfg, update_config
from models.TAN import MultiScale_TAN
from torch.utils.data import DataLoader
from datasets.charades import CharadesSampleCG
from datasets import train_collate_fn, train_sample_collate_fn
from datasets.utils import recover_to_single_map, nms_acc, miou


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--load', help='load checkpoint for evaluate', required=True, type=str)
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
def eval_log(eval_acc, eval_miou, desc):
    eval_acc = eval_acc.flatten().tolist()
    eval_miou = eval_miou.item()
    log = "Rank1@0.5={0:05.3%}, Rank1@0.7={1:05.3%}, mIoU={4:05.3%}" \
          "\nRank5@0.5={2:05.3%}, Rank5@0.7={3:05.3%}\n".format(*eval_acc, eval_miou)
    print(log)
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

if __name__ == '__main__':
    torch.manual_seed(1111)
    torch.cuda.manual_seed(11111)
    args = parse_args()
    reset_config(cfg, args)
    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    tt_data = CharadesSampleCG(cfg.DATASET, 'test_trivial')
    nw_data = CharadesSampleCG(cfg.DATASET, 'novel_word')
    nc_data = CharadesSampleCG(cfg.DATASET, 'novel_composition')

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
    model.load_state_dict(torch.load(args.load))

    model.eval()
    tt_score = eval_model(model, tt_loader, 'TT')
    nw_score = eval_model(model, nw_loader, 'NW')
    nc_score = eval_model(model, nc_loader, 'NC')
