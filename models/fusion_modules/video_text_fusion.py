import torch
from torch import nn
import torch.nn.functional as F

class MulNormFusion(nn.Module):

    def __init__(self, cfg):
        super(MulNormFusion, self).__init__()
        self.vis_conv = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        torch.nn.init.kaiming_normal_(self.vis_conv.weight)
        torch.nn.init.zeros_(self.vis_conv.bias)

    def forward(self, txt_h, map_h, video_mask):
        map_h = self.vis_conv(map_h)
        map_h = self.dropout1(map_h)
        fused_h = F.normalize(txt_h * map_h) * video_mask
        return self.dropout2(fused_h)