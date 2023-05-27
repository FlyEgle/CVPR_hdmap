import cv2
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile



@PIPELINES.register_module()
class LoadMultiViewImageFromFilesForAv2(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        # TODO: crop or resize to same size
        # if 's3' in filename[0]:
        #     img = np.stack([mmcv.imresize(self._imread(name)) for name in filename], axis=-1)
        # else:
        #     img = np.stack(
        #         [mmcv.imresize(mmcv.imread(name, self.color_type)) for name in filename], axis=-1)

        if 's3' in filename[0]:
            img = [self._imread(name) for name in filename]
        else:
            img = [mmcv.imread(name, self.color_type) for name in filename]

        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)

        if self.to_float32:
            img = [ i.astype(np.float32) for i in img]
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        # results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img']  = img 
        # results['img_shape'] = img.shape
        # results['ori_shape'] = img.shape
        # # Set initial values for default meta_keys
        # results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
    
    def _imread(self, path):
        from petrel_client.client import Client
        
        client = Client("~/petreloss.conf") # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
        img_bytes = client.get(path)
        assert(img_bytes is not None)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # import os
        # mmcv.mkdir_or_exist(path.split('/')[-5])
        # cv2.imwrite(os.path.join(path.split('/')[-5], f"{path.split('/')[-2]}_{path.split('/')[-1]}"), img)

        return img

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str