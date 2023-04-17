import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import cv2
from mmdet.datasets.pipelines import to_tensor

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor


        # # ===== 可视化
        # import cv2
        # import os
        # for index, img in enumerate(results['img']):
        #     img = np.uint8(img)
        #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     lidar2img = results['lidar2img'][index]

        #     for instance in results['ann_info']['gt_instance_xyz_list']:
        #         # instance = np.concatenate([instance, np.zeros((instance.shape[0], 1))], axis=1) 
        #         instance = np.concatenate([instance, np.ones((instance.shape[0], 1))], axis=1)
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
        #     CAM_TYPE = ['ring_front_center', 'ring_front_right', 'ring_front_left', 'ring_rear_right', 'ring_rear_left', 'ring_side_right', 'ring_side_left']
        #     dir = f"../padding_img/{results['scene_token']}/{results['sample_idx']}"
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
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgbf
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus','lidar2global',
                            'camera2ego','camera_intrinsics','img_aug_matrix','lidar2ego',
                            'cam_extrinsics'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'



@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
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
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        img_aug_matrix = [scale_factor for _ in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_aug_matrix'] = img_aug_matrix
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
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
            lidar2img = results['lidar2img'][index]
            for instance in results['ann_info']['gt_instance_xyz_list']:
                # instance = np.concatenate([instance, np.zeros((instance.shape[0], 1))], axis=1) 
                instance = np.concatenate([instance, np.ones((instance.shape[0], 1))], axis=1)
               
                instance = instance @ lidar2img.T
                instance = instance[instance[:, 2] > 1e-5]
                points_2d = instance[:, :2] / instance[:, 2:3]
                points_2d = points_2d.astype(int)
                
                gt_uvsegmentation = cv2.polylines(gt_uvsegmentation, points_2d[None], False, (255, 255, 255), thickness=self.thickness)

            # # 过滤掉 padding 的那部分区域 
            # gt_uvsegmentation[results['before_pad_shape'][index][0]:] = 0
            # gt_uvsegmentation[:, results['before_pad_shape'][index][1]:] =0
            gt_uvsegmentation = gt_uvsegmentation.astype(np.float64)
            gt_uvsegmentation = gt_uvsegmentation / 255.0 
            gt_uvsegmentations.append(gt_uvsegmentation)
            
            # # # 可视化 gt_uvsegmentation 
            # import os
            # CAM_TYPE = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']
            # dir = f"../gt_uvsegmentations_raw/{results['scene_token']}/{results['sample_idx']}"
            # mmcv.mkdir_or_exist(dir)
            # cv2.imwrite(os.path.join(dir, f"{CAM_TYPE[index]}.png" ), gt_uvsegmentation)

        results['gt_uvsegmentations'] = DC(to_tensor(gt_uvsegmentations), stack=True) 

        # results['gt_uvsegmentations'] = gt_uvsegmentations
       
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


from shapely.geometry import LineString, box
import cv2
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC
@PIPELINES.register_module()
class GenerateBEVSegmentationForArgo(object):
    """GenerateBEVSegmentationForArgo.

   GenerateBEVSegmentationForArgo.

    Args:
        map_size (list, optional):
            Defaults to [(-50, 50), (-25, 25)].
        bev_size (tuple, optional): (bev_size_h, bev_size_w).
            Defaults to (100, 200).
        thickness (int, optional): thickness for drawing SDMap path.
            Defaults to 2.
    """

    def __init__(self,
                 map_size=[(-30, 30), (-15, 15)],  # [(min_h, max_h), (min_w, max_w)]
                 bev_size=(100, 200),  # H, W
                 thickness=2, 
              ):
        self.bev_size = bev_size
        # 注意这里 除的顺序
        self.scale = ((map_size[0][1] - map_size[0][0]) / bev_size[1], (map_size[1][1] - map_size[1][0]) / bev_size[0]) 
        self.map_size = map_size
        self.patch_size = box(map_size[0][0], map_size[1][0], map_size[0][1], map_size[1][1])   # (x_min, y_min, x_max, y_max)

        self.thickness = thickness


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded SDMap Pts and Patch.
        """
        gt_instances = results['ann_info']['gt_instance_xyz_list']
        bev_map_patch = np.zeros(self.bev_size, dtype=np.uint8)
        bev_map_lines = []
        
        for pts in gt_instances:
            new_pts = LineString(pts[:,:2])
            new_pts = new_pts.intersection(self.patch_size)
            if new_pts.geom_type == 'MultiLineString':
                for new_pts_single in new_pts.geoms:
                    if new_pts_single.length == 0.0:
                        continue
                    line = (np.asarray(list(new_pts_single.coords)) + np.array([self.map_size[0][1], self.map_size[1][1]])) / self.scale
                    line = line.astype(np.int)

                    cv2.polylines(bev_map_patch, line[None], False, (255, 255, 255), thickness=self.thickness)

            elif new_pts.length != 0.0:
                line = (np.asarray(list(new_pts.coords)) + np.array([self.map_size[0][1], self.map_size[1][1]])) / self.scale
                line = line.astype(np.int)
              
                cv2.polylines(bev_map_patch, line[None], False, (255, 255, 255), thickness=self.thickness)
        
        bev_map_patch = cv2.flip(bev_map_patch, 0)

        # # 可视化
        # vis_dir = '../vis_sdmap/'
        # mmcv.mkdir_or_exist(vis_dir)
        # import os
        # cv2.imwrite(os.path.join(vis_dir, f"{results['sample_idx']}.png"), bev_map_patch)
      
        bev_map_patch = bev_map_patch.astype(np.float)
        bev_map_patch /= bev_map_patch.max()
        results['gt_bevsegmentations'] = DC(to_tensor([bev_map_patch]), stack=True) 
        return results


    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}map_size={self.map_size}, '
        repr_str += f'{indent_str}bev_size={self.bev_size}, '
        repr_str += f'{indent_str}scale={self.scale}, '
        repr_str += f'{indent_str}thickness={self.thickness}, '
        return repr_str