from torch import nn
import models.clip_modules as clip_modules
import models.prop_modules as prop_modules
import models.lang_modules as lang_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
import models.pred_modules as pred_modules

class MultiScale_TAN(nn.Module):

    def __init__(self, cfg):
        super(MultiScale_TAN, self).__init__()
        self.cfg = cfg
        self.clip_module = getattr(clip_modules, cfg.CLIP_MODULE.NAME)(cfg.CLIP_MODULE.PARAMS)
        self.prop_module = getattr(prop_modules, cfg.PROP_MODULE.NAME)(cfg.PROP_MODULE.PARAMS)
        self.lang_module = getattr(lang_modules, cfg.FUSION_MODULE.NAME)(cfg.FUSION_MODULE.PARAMS)
        self.fusion_modules = nn.ModuleList()
        self.map_modules = nn.ModuleList()
        self.pred_modules = nn.ModuleList()

        for _ in range(cfg.PARAMS.NUM_SCALES):
            self.fusion_modules.append(getattr(fusion_modules, cfg.FUSION_MODULE.NAME)(cfg.FUSION_MODULE.PARAMS))
            self.map_modules.append(getattr(map_modules, cfg.MAP_MODULE.NAME)(cfg.MAP_MODULE.PARAMS))
            self.pred_modules.append(getattr(pred_modules, cfg.PRED_MODULE.NAME)(cfg.PRED_MODULE.PARAMS))

    def forward(self, sent_feat, sent_mask, video_feat, video_masks):
        vis_h = self.clip_module(video_feat)
        prop_hs = self.prop_module(vis_h, video_masks)
        txt_h = self.lang_module(sent_feat, sent_mask)

        predictions = []
        for idx, (prop_h, video_mask) in enumerate(zip(prop_hs, video_masks)):
            video_mask = video_mask.unsqueeze(1)
            fused_h = self.fusion_modules[idx](txt_h, prop_h, video_mask)
            map_h = self.map_modules[idx](fused_h, video_mask)
            prediction = self.pred_modules[idx](map_h, video_mask)
            predictions.append(prediction)
        return predictions
