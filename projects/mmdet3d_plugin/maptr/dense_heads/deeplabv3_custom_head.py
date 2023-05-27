

# from mmseg.registry import MODELS
from mmseg.ops import resize
from mmseg.models.losses import accuracy
from mmseg.models.builder import build_loss, HEADS
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32

import torch
import torch.nn as nn


@HEADS.register_module()
class DeepLabV3CustomHead(DepthwiseSeparableASPPHead):
    def __init__(self, 
                 downsample_label_ratio=1.0,
                 loss_decode_custom=None,
                 loss_name='',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsample_label_ratio = downsample_label_ratio
        self.loss_name = loss_name
        if isinstance(loss_decode_custom, dict):
            self.loss_decode_custom = build_loss(loss_decode_custom)
        elif isinstance(loss_decode_custom, (list, tuple)):
            
            loss_decode_custom_dict = dict()
            for loss in loss_decode_custom:
                loss_decode_custom_dict[loss['loss_name']] = build_loss(loss['loss'])
            
            self.loss_decode_custom = nn.ModuleDict(
               loss_decode_custom_dict 
            )
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode_custom)}')

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        if self.downsample_label_ratio > 0:
            seg_label = seg_label.float()
            target_size = (int(seg_label.shape[2] // self.downsample_label_ratio),
                           int(seg_label.shape[3] // self.downsample_label_ratio))
            seg_label = resize(
                input=seg_label, size=target_size, mode='nearest')
            seg_label = seg_label.long()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        for loss_decode in self.loss_decode_custom.keys():
            loss[self.loss_name+'_'+loss_decode] = self.loss_decode_custom[loss_decode](
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    )

        loss[self.loss_name+'_'+'acc_seg'] = accuracy(
            seg_logit, seg_label)
        loss[self.loss_name+'_'+'iou'] = get_batch_iou(seg_logit, seg_label )
        return loss


def get_batch_iou(pred_map, gt_map):

    assert pred_map.ndim == gt_map.ndim + 1
    assert pred_map.size(0) == gt_map.size(0)
    pred_value, pred_label = pred_map.topk(1, dim=1)
    
    gt_map = gt_map.unsqueeze(1)
    # draw_seg(pred_label, gt_map)
    pred_map = pred_label.bool()
    gt_map = gt_map.bool()

    intersect = (pred_map & gt_map).sum().float()
    union = (pred_map | gt_map).sum().float()
    iou = intersect / (union + 1e-7) 
    return iou


IOU_INDEX=0
def draw_seg(pred_map, gt_map):
    import cv2
    import mmcv
    import numpy as np
    global IOU_INDEX
    mmcv.mkdir_or_exist('iou')
    for pred, gt in zip(pred_map, gt_map):
        pred = (pred.cpu().numpy().astype(np.int8) * 255).astype(np.uint8)
        gt = (gt.cpu().numpy() * 255).astype(np.uint8)
        res = np.hstack([pred, gt])
        res = np.squeeze(res)
        cv2.imwrite(f'./iou/seg_{IOU_INDEX}.png', res)
        IOU_INDEX += 1