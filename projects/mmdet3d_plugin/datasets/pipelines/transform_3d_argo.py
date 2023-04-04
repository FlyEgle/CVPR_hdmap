import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import cv2
import os


@PIPELINES.register_module()
class RandomScaleImageMultiViewImageArgo(object):
    """Random scale the image for Argo
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_scale = np.random.choice(self.scales)

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        # lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]    # TODO: 这里的符号需要改变 ego2img
        ego2img = [scale_factor @ l2i for l2i in results['ego2img']] 
        cam_intrinsic = [scale_factor @ c2i for c2i in results['camera_intrinsics']]

        # results['lidar2img'] = lidar2img
        results['ego2img'] = ego2img
        results['camera_intrinsics'] = cam_intrinsic
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        results['scale_factor'] = [rand_scale for _ in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class CropFrontViewImageForArgo(object):

    def __init__(self, crop_h=(356, 1906)):
        self.crop_h = crop_h

    def _crop_img(self, results):
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'][0] = results['img'][0][self.crop_h[0]:self.crop_h[1]]
        results['img_shape'] = [img.shape for img in results['img']]
        results['crop_shape'][0] = np.array([0, self.crop_h[0]])


    def _crop_cam_intrinsic(self, results):
        results['camera_intrinsics'][0][1, 2] -= self.crop_h[0]
        cam_intrinsic = results['camera_intrinsics'][0]
        viewpad = np.eye(4)
        viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
        results['ego2img'][0] = viewpad @ results['camera_extrinsics'][0]

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._crop_img(results)
        self._crop_cam_intrinsic(results)
        return results

@PIPELINES.register_module()
class PadMultiViewImageForArgo(object):

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        results['before_pad_shape'] = [img.shape for img in results['img']]
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor
       

        # # 可视化 gt
        # import cv2
        # for index, img in enumerate(results['img']):
        #     img = np.uint8(img)
        #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     lidar2img = results['ego2img'][index]

        #     for instance in results['gt_bboxes_3d'].instance_list:
        #         instance = np.array(instance.coords)
        #         instance = np.concatenate([instance, np.zeros((instance.shape[0], 1))], axis=1)
        #         instance = np.concatenate([instance, np.ones((instance.shape[0], 1))], axis=1)

        #     # for instance in results['gt_for_uvsegmentation_list']:
        #     #     # instance = np.concatenate([instance, np.zeros((instance.shape[0], 1))], axis=1) 
        #     #     instance = np.concatenate([instance, np.ones((instance.shape[0], 1))], axis=1)
        #         instance = instance @ lidar2img.T
        #         instance = instance[instance[:, 2] > 1e-5]

        #         # if xyz1.shape[0] == 0:
        #         #     continue
        #         points_2d = instance[:, :2] / instance[:, 2:3]
        #         # mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])
        #         points_2d = points_2d.astype(int)
                
        #         img = cv2.polylines(img, points_2d[None], False, (0, 255, 0), 2)

        #     # mmcv.mkdir_or_exist('instance_draw')
        #     # import os
        #     # cv2.imwrite(os.path.join('instance_draw', f'{index}.png'), img)

        #     CAM_TYPE = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']
        #     dir = f"../padding_img_raw/{results['scene_token']}/{results['sample_idx']}"
        #     mmcv.mkdir_or_exist(dir)
        #     cv2.imwrite(os.path.join(dir, f"{CAM_TYPE[index]}.png" ), img)


    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str



@PIPELINES.register_module()
class GenerateUVSegmentationForArgo(object):

    def __init__(self, thickness=10):
        self.thickness = thickness

    def _generate_uvsegmentation(self, results):
        """generate uvsegmentation."""
        gt_uvsegmentations = []
        for img in results['img']:
            gt_uvsegmentation = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for index, img in enumerate(results['img']):
            gt_uvsegmentation = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            lidar2img = results['ego2img'][index]

            for instance in results['gt_for_uvsegmentation_list']:
                # instance = np.concatenate([instance, np.zeros((instance.shape[0], 1))], axis=1) 
                instance = np.concatenate([instance, np.ones((instance.shape[0], 1))], axis=1)
               
                instance = instance @ lidar2img.T
                instance = instance[instance[:, 2] > 1e-5]
                points_2d = instance[:, :2] / instance[:, 2:3]
                points_2d = points_2d.astype(int)
                
                gt_uvsegmentation = cv2.polylines(gt_uvsegmentation, points_2d[None], False, (255, 255, 255), thickness=self.thickness)

            # 过滤掉 padding 的那部分区域 
            gt_uvsegmentation[results['before_pad_shape'][index][0]:] = 0
            gt_uvsegmentation[:, results['before_pad_shape'][index][1]:] =0
      
            gt_uvsegmentations.append(gt_uvsegmentation.astype(np.float64))
            
            # # # 可视化 gt_uvsegmentation 
            # CAM_TYPE = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']
            # dir = f"../gt_uvsegmentations_raw/{results['scene_token']}/{results['sample_idx']}"
            # mmcv.mkdir_or_exist(dir)
            # cv2.imwrite(os.path.join(dir, f"{CAM_TYPE[index]}.png" ), gt_uvsegmentation)


        results['gt_uvsegmentations'] = gt_uvsegmentations
       
    def __call__(self, results):
        """Call function to generate uvsegmentation gt for aux-head-loss.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._generate_uvsegmentation(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'thickness={self.thickness}, '
        return repr_str




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
            results['camera_intrinsics'][index] = scale_factor @ results['camera_intrinsics'][index] 
            results['ego2img'][index] = scale_factor @ results['ego2img'][index]
        
        return results