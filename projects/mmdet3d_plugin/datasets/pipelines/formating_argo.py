
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
import copy



@PIPELINES.register_module()
class ArgoFormatBundle3D(object):
    """Custom formatting bundle for 3D Lane.
    """

    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        # super(ArgoFormatBundle3D, self).__init__(class_names, **kwargs)

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'gt_labels_3d' in results:
            results['gt_labels_3d'] = DC(results['gt_labels_3d'], cpu_only=False)

        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'] = DC(results['gt_bboxes_3d'], cpu_only=True)

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)

        if 'ego2img' in results:        # argo 数据集中只有img、cam、ego、global坐标系，而nuscenes多了一个lidar坐标系
            results['lidar2img'] = copy.deepcopy(results['ego2img'])

        if 'ego2global' in results:
            results['lidar2global'] = copy.deepcopy(results['ego2global'])

        
        results['ego2img'] = DC(to_tensor(results['ego2img']), stack=True)

        if 'gt_uvsegmentations' in results:
            results['gt_uvsegmentations'] = DC(to_tensor(results['gt_uvsegmentations']), stack=True) 


        # if 'gt_lanes_3d' in results:
        #     results['gt_lanes_3d'] = DC(
        #         to_tensor(results['gt_lanes_3d']))
        # if 'gt_lane_labels_3d' in results:
        #     results['gt_lane_labels_3d'] = DC(
        #         to_tensor(results['gt_lane_labels_3d']))
        # if 'gt_lane_adj' in results:
        #     results['gt_lane_adj'] = DC(
        #         to_tensor(results['gt_lane_adj']))
        # if 'gt_lane_lcte_adj' in results:
        #     results['gt_lane_lcte_adj'] = DC(
        #         to_tensor(results['gt_lane_lcte_adj']))

        # results = super(ArgoFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        # repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
