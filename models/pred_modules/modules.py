import torch
from torch import nn

class ProposalHead(nn.Module):
    def __init__(self, cfg):
        super(ProposalHead, self).__init__()
        self.predictor1 = nn.Conv2d(cfg.INPUT_SIZE, cfg.INPUT_SIZE // 2, 1, 1)
        self.relu1 = nn.ReLU()
        self.predictor2 = nn.Conv2d(cfg.INPUT_SIZE // 2, cfg.INPUT_SIZE // 4, 1, 1)
        self.relu2 = nn.ReLU()
        self.predictor3 = nn.Conv2d(cfg.INPUT_SIZE // 4, getattr(cfg, 'OUTPUT_SIZE', 1), 1, 1)

        torch.nn.init.kaiming_normal_(self.predictor1.weight)
        torch.nn.init.zeros_(self.predictor1.bias)
        torch.nn.init.kaiming_normal_(self.predictor2.weight)
        torch.nn.init.zeros_(self.predictor2.bias)
        torch.nn.init.kaiming_normal_(self.predictor3.weight)
        torch.nn.init.zeros_(self.predictor3.bias)

    def forward(self, vis_input, video_mask):
        output = self.relu1(self.predictor1(vis_input))
        output = self.relu2(self.predictor2(output))
        return self.predictor3(output).squeeze(1)