import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import cv2
import os


@PIPELINES.register_module()
class ResizeMultiViewImageForArgo(object):

    def __init__(self, resize=(2048, 1550)):
        self.size = resize

    def __call__(self, results):
        """Call function to resize img.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        for index, img in enumerate(results['img']):
            results['img'][index], w_scale, h_scale = mmcv.imresize(img, (self.size), return_scale=True)
            scale_factor = np.eye(4)
            scale_factor[0, 0] *= w_scale
            scale_factor[1, 1] *= h_scale
            results['cam_intrinsics'][index] = scale_factor @ results['cam_intrinsics'][index] 
            results['lidar2img'][index] = scale_factor @ results['lidar2img'][index]
        
        return results