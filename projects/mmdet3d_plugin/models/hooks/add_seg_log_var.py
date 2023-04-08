from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import is_module_wrapper

import torch

@HOOKS.register_module()
class AddSegmentationLogVarHook(Hook):
    def __init__(self, interval=1):
        pass

    def after_train_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if model.uvsegmentations_aux_head is not None:
            iou = model.uvsegmentations_aux_head.iou
            runner.log_buffer.update(iou)
        