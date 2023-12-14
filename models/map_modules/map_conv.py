import torch
import torch.nn as nn

class MapGatedConv(nn.Module):
    def __init__(self, cfg):
        super(MapGatedConv, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList()
        assert len(cfg.HIDDEN_SIZES) == len(cfg.KERNEL_SIZES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.STRIDES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.PADDINGS) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.DILATIONS)

        in_channels = [cfg.INPUT_SIZE] + cfg.HIDDEN_SIZES[:-1]
        out_channels = cfg.HIDDEN_SIZES

        for i, (k, s, p, d) in enumerate(zip(cfg.KERNEL_SIZES, cfg.STRIDES, cfg.PADDINGS, cfg.DILATIONS)):
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(in_channels[i], out_channels[i], k, s, p, d),
                nn.Conv2d(in_channels[i], 1, k, s, p, d),
            ]))
        
        for block in self.blocks:
            torch.nn.init.xavier_normal_(block[1].weight)
            torch.nn.init.zeros_(block[1].bias)
            torch.nn.init.kaiming_normal_(block[0].weight)
            torch.nn.init.zeros_(block[0].bias)

    def forward(self, x, video_mask):
        for block in self.blocks:
            gate = torch.sigmoid(block[1](x))  # [B, 1, H, W]
            x = torch.relu(block[0](x))  # [B, C, H, W]
            x = gate * x
        return x