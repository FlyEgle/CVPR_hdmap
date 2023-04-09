"""Generate DataSet for av2 dataset
"""
import os 
import copy
import mmcv
import json
import torch
import random
import tempfile
import numpy as np

from os import path as osp
from shapely import affinity, ops
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.nuscenes_dataset import *

from .nuscenes_dataset import CustomNuScenesDataset, NuScenesDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString, Polygon
import cv2
from .av2_dataset import CustomAV2MapDataset
from .evaluation_argo.vector_eval import VectorEvaluate



def add_rotation_noise(extrinsics, std=0.01, mean=0.0):
    #n = extrinsics.shape[0]
    noise_angle = torch.normal(mean, std=std, size=(3,))
    # extrinsics[:, 0:3, 0:3] *= (1 + noise)
    sin_noise = torch.sin(noise_angle)
    cos_noise = torch.cos(noise_angle)
    rotation_matrix = torch.eye(4).view(4, 4)
    #  rotation_matrix[]
    rotation_matrix_x = rotation_matrix.clone()
    rotation_matrix_x[1, 1] = cos_noise[0]
    rotation_matrix_x[1, 2] = sin_noise[0]
    rotation_matrix_x[2, 1] = -sin_noise[0]
    rotation_matrix_x[2, 2] = cos_noise[0]

    rotation_matrix_y = rotation_matrix.clone()
    rotation_matrix_y[0, 0] = cos_noise[1]
    rotation_matrix_y[0, 2] = -sin_noise[1]
    rotation_matrix_y[2, 0] = sin_noise[1]
    rotation_matrix_y[2, 2] = cos_noise[1]

    rotation_matrix_z = rotation_matrix.clone()
    rotation_matrix_z[0, 0] = cos_noise[2]
    rotation_matrix_z[0, 1] = sin_noise[2]
    rotation_matrix_z[1, 0] = -sin_noise[2]
    rotation_matrix_z[1, 1] = cos_noise[2]

    rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

    rotation = torch.from_numpy(extrinsics.astype(np.float32))
    rotation[:3, -1] = 0.0
    # import pdb;pdb.set_trace()
    rotation = rotation_matrix @ rotation
    extrinsics[:3, :3] = rotation[:3, :3].numpy()
    return extrinsics


def add_translation_noise(extrinsics, std=0.01, mean=0.0):
    # n = extrinsics.shape[0]
    noise = torch.normal(mean, std=std, size=(3,))
    extrinsics[0:3, -1] += noise.numpy()
    return extrinsics



@DATASETS.register_module()
class CustomAV2MapDatasetGather(CustomAV2MapDataset):
    def __init__(self,
                ann_file_s3=None,
                map_ann_file=None,
                 *args, 
                 **kwargs,
                 ):
        super(CustomAV2MapDatasetGather, self).__init__(ann_file_s3=ann_file_s3, map_ann_file=map_ann_file, *args, **kwargs)
        # self.data_infos = self._load_annotations(map_ann_file) # debug
        # self._load_annotations_s3(ann_file_s3)
    
    def _load_annotations_s3(self, ann_file_s3):
        self.ann_file_s3 = ann_file_s3
        self.data_infos_s3 = None
        if self.ann_file_s3 is not None:
            data_infos_list = [ mmcv.load(single_ann_file_s3, file_format='pkl') for single_ann_file_s3 in self.ann_file_s3] 
            data_infos = [] 
            for singel_data_infos in data_infos_list:
                 
                if isinstance(singel_data_infos, dict):
                    data_infos.extend(list(singel_data_infos.values()))
                else:
                    data_infos.extend(singel_data_infos) 
            self.data_infos_s3 = data_infos


    def _load_annotations(self, ann_file):
        # return super().load_annotations(ann_file)
        """Load annotations form ann_file.

        Args:
            ann_file (str): Path of the annotation file.
        
        Returns:
            list[dict] : List of each samples annotation

        """
        self.ann_file = ann_file
        ann_list = [mmcv.load(single_ann_file) for single_ann_file in ann_file ]
        samples = []
        for ann in ann_list:
            for seg_id, sequence in ann.items():
                samples.append(sequence)
        
        samples_list = []
        # timestamp instead of the frame_token
        for idx, scene in enumerate(samples):
            scene_list = []
            for frame_idx in range(len(scene)):
                # first frame
                # print(frame)
                frame = scene[frame_idx]
                if frame_idx == 0:
                    frame["frame_idx"] = frame_idx
                    frame["prev"] = ""
                    frame["next"] = scene[frame_idx+1]["timestamp"]
                # last frame
                elif frame_idx == len(scene) - 1:
                    frame["frame_idx"] = frame_idx
                    frame["prev"] = scene[frame_idx - 1]["timestamp"]
                    frame["next"] = ""
                else:
                    frame["frame_idx"] = frame_idx
                    frame["prev"] = scene[frame_idx - 1]["timestamp"]
                    frame["next"] = scene[frame_idx + 1]["timestamp"]

                frame["lidar2ego_rotation"] = [
                    [1, 0, 0], [0, 1, 0], [0, 0, 1]
                ]
                frame["lidar2ego_translation"] = [0, 0, 0]

                scene_list.append(frame)

            samples_list.extend(scene_list)

        return samples_list
    
    def get_data_info(self, index):
        """Get data info according to the given index
        
        Args:
            index (int): Index of the smaple data to get.
        
        Returns: follow the nuscens format
             dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.

        """
        # sample
        info = self.data_infos[index]
        input_dict = dict(
            token                  = info["segment_id"] + '_' + info["timestamp"],
            sample_idx             = info['timestamp'],   # timestamp == token, only one
            ego2global_translation = info["pose"]["ego2global_translation"],
            ego2global_rotation    = info["pose"]["ego2global_rotation"],
            lidar2ego_translation  = info["lidar2ego_translation"],
            lidar2ego_rotation     = info["lidar2ego_rotation"],
            timestamp              = info["timestamp"],
            prev_idx               = info["prev"],
            next_idx               = info["next"],
            scene_token            = info["segment_id"],
            frame_idx              = info["frame_idx"],
            can_bus                = np.zeros((18, )),
          #  lidar_anno             = info["annotation"],   # {"ped_crossing": list[(nx4)], "divider": list[nx4], "boundary": list[nx4]}
            pts_filename           = None,
            sweeps                 = None,
            map_location           = None,

        )
        # if not self.test_mode:
        input_dict['lidar_anno'] = info["annotation"] 

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = info["lidar2ego_rotation"]
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        # print("lidar2ego": lidar2ego)
        input_dict["lidar2ego"] = lidar2ego
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_extrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            # input_dict["cam_extrinsics"] =  []
            for cam_type, cam_info in info["sensor"].items():
                if self.ann_file_s3 is None:                # by shengyin
                    folder = self.map_ann_file[0].split("/")[-1].split("_")[0]
                    prefix = self.map_ann_file[0].split("/OpenLaneV2")[0] + "/OpenLaneV2"  # hard code
                    prefix_path = os.path.join(prefix, folder)
                    image_path = os.path.join(prefix_path, cam_info["image_path"])
                else:
                    image_path = self.data_infos_s3[index]['sensor'][cam_type]['image_path'].replace(
                        f"{self.data_infos_s3[index]['segment_id']}/image", f"{self.data_infos_s3[index]['meta_data']['source_id']}/sensors/cameras", 1)
                
                    image_path = os.path.join('s3://odl-flat/Argoverse2/Sensor_Dataset/sensor', image_path)
                image_paths.append(image_path)

                # TODO: this is hard code, need modify 
                # mmcv.imread -> (h, w, c)
                
                # if self.ann_file_s3 is not None:
                #     img_height, img_width, _ = self._imread(image_path).shape

                # else:
                #     img_height, img_width, _ = mmcv.imread(os.path.join(prefix_path,
                #                                                     cam_info["image_path"])).shape 
               
                # constant_resize_shape = (1600, 900)
                # constant_resize_shape = (2048, 1550)

                # resize_ratio = [constant_resize_shape[0] / img_width, 
                # constant_resize_shape[1] / img_height]

                # obtain lidar to image transformation matrix, av2 is ego2cam
                ego2cam = cam_info["extrinsic"]     # 4x4 eog->cam
                # print(ego2cam)
                cam_extrinsics.append(ego2cam)
                ego2cam = np.array(ego2cam)
                lidar2cam_r = ego2cam[:3, :3]  # 3x3
                lidar2cam_t = ego2cam[:3, 3]
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r
                lidar2cam_rt[:3, 3] = lidar2cam_t
                """
                    [
                        [cosx, 0, 0, tranx],
                        [0, cosy, 0, trany],
                        [0, 0, cosz, tranz],
                        [0, 0, 0, 1]
                    ]

                lidar2cam_t = [tranx, trany, tranz]

                """
                lidar2cam_rt_t = lidar2cam_rt.T
                """
                        [
                        [cosx, 0, 0, 0],
                        [0, cosy, 0, 0],
                        [0, 0, cosz, 0],
                        [tranx, trany, tranz, 1]
                    ]

                """

                if self.noise == 'rotation':
                    lidar2cam_rt_t = add_rotation_noise(lidar2cam_rt_t, std=self.noise_std)
                elif self.noise == 'translation':
                    lidar2cam_rt_t = add_translation_noise(
                        lidar2cam_rt_t, std=self.noise_std)
                
                # lidar/ego 2 uv
                intrinsic = np.array(cam_info['intrinsic'])
                # change intrinsic beof the image size have been change
                # intrinsic = intrinsic_project(resize_ratio, intrinsic) # TODO: hard code
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                cam2lidar_r = np.linalg.inv(lidar2cam_r).T
                cam2lidar_t = -lidar2cam_t @ cam2lidar_r
                camera2ego[:3, :3] = cam2lidar_r
                camera2ego[:3, 3]  = cam2lidar_t

                input_dict["camera2ego"].append(camera2ego)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = intrinsic
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filename   = image_paths,
                    lidar2img      = lidar2img_rts,
                    cam_intrinsics = cam_intrinsics,
                    lidar2cam      = lidar2cam_rts,
                    cam_extrinsics =  cam_extrinsics
                    
                )
            )
                
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        # pose & angle
        # print(input_dict["ego2global_rotation"])
        # TODO:DBEUG
        
        rotation = Quaternion(matrix=np.array(input_dict['ego2global_rotation']))
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']  # default is the zeros 
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        # lidar2global is same with the ego2global in av2
        lidar2global = np.eye(4)
        lidar2global[:3, :3] = rotation.rotation_matrix
        lidar2global[:3, 3]  = translation
        input_dict["lidar2global"] = lidar2global

        return input_dict
    


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        self.evaluator = VectorEvaluate(self.ann_file[0])

        # 提交成绩
        # submisson_vector_path = result_files['pts_bbox']
        # submisson_vector = mmcv.load(submisson_vector_path)
        # submisson_vector['meta'] = {
        #         'use_lidar': False,
        #         'use_camera': True,
        #         "use_external": False,                     
        #         "output_format": "vector",                  
        #         'use_external': False,

        #         # NOTE: please fill the information below
        #         'method': 'maptr',                            
        #         'authors': ['JiangShengyin'],                          
        #         'e-mail': 'shengyin@bupt.edu.cn',                            
        #         'institution / company': 'bupt',         
        #         'country / region': 'china',                  
        # }
        # mmcv.dump(submisson_vector, 'submisson_vector.json')


        results_dict = self.evaluator.evaluate(result_files['pts_bbox'], logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict