import torch
from torch import nn


class MulNormFusion(nn.Module):
    def __init__(self, cfg):
        super(MulNormFusion, self).__init__()
        self.textual_encoder = getattr(nn, cfg.TXT_ENCODER.NAME)(
            cfg.TXT_INPUT_SIZE, cfg.TXT_HIDDEN_SIZE//2 if cfg.TXT_ENCODER.BIDIRECTIONAL else cfg.TXT_HIDDEN_SIZE,
            num_layers=cfg.TXT_ENCODER.NUM_LAYERS, bidirectional=cfg.TXT_ENCODER.BIDIRECTIONAL, batch_first=True
        )
        self.tex_linear1 = nn.Linear(cfg.TXT_HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.relu1 = torch.nn.ReLU()
        self.tex_linear2 = nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.relu2 = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        for name, param in self.textual_encoder.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        torch.nn.init.xavier_normal_(self.tex_linear1.weight)
        torch.nn.init.zeros_(self.tex_linear1.bias)
        torch.nn.init.kaiming_normal_(self.tex_linear2.weight)
        torch.nn.init.zeros_(self.tex_linear2.bias)

    def forward(self, sent_feat, sent_mask):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(sent_feat)[0] * sent_mask  # [B, L, H]
        txt_h = torch.sum(txt_h, dim=1) / torch.sum(sent_mask, dim=1)
        txt_h = self.dropout(txt_h)
        txt_h = self.relu1(self.tex_linear1(txt_h))
        txt_h = self.relu2(self.tex_linear2(txt_h))[:,:,None,None]
        return txt_h

