import torch
from torch import nn
import torch.nn.functional as F

class ClipAvgPool(nn.Module):

    def __init__(self, cfg):
        super(ClipAvgPool, self).__init__()
        self.vis_conv = nn.Conv1d(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)

        torch.nn.init.kaiming_normal_(self.vis_conv.weight)
        torch.nn.init.zeros_(self.vis_conv.bias)

    def forward(self, video_feat):
        vis_h = torch.relu(self.vis_conv(video_feat))
        return vis_h
