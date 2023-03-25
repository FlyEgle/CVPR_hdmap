import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
import os
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random

from .nuscenes_dataset import CustomNuScenesDataset
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from mmdet.datasets.pipelines import to_tensor
import json

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

class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates

    """
    def __init__(self, 
                 instance_line_list, 
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
                                dtype=torch.float32)
        instance_se_points_tensor[:,0] = torch.clamp(instance_se_points_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,1] = torch.clamp(instance_se_points_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_se_points_tensor[:,2] = torch.clamp(instance_se_points_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,3] = torch.clamp(instance_se_points_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
                            dtype=torch.float32)
        instance_bbox_tensor[:,0] = torch.clamp(instance_bbox_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,1] = torch.clamp(instance_bbox_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_bbox_tensor[:,2] = torch.clamp(instance_bbox_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,3] = torch.clamp(instance_bbox_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0,2,1)
            sampled_pts = torch.nn.functional.interpolate(poly_pts,size=(self.fixed_num),mode='linear',align_corners=True)
            sampled_pts = sampled_pts.permute(0,2,1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list,dim=0)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num-shift_pts.shape[0],pts_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape
            if shifts_num > 2*final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index+shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts,flip1_shifts_pts),axis=0)
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            if multi_shifts_pts_tensor.shape[0] < 2*final_shift_num:
                padding = torch.full([final_shift_num*2-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num*2, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num*2-shift_pts.shape[0],pts_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False

        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor



@DATASETS.register_module()
class AV2Dataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This datset add static map elements
    """
    MAPCLASSES = ('ped_crossing', 'divider', 'boundary',)
    def __init__(self,
                 map_ann_file=None, 
                 queue_length=4, 
                 bev_size=(200, 200), 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False, 
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 noise='None',
                 noise_std=0,
                 ann_file_s3=None,      # by shengyin
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_ann_file = map_ann_file

        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.is_vis_on_test = False
        self.noise = noise
        self.noise_std = noise_std

        self.cat2id = {
            'ped_crossing': 0,
            'divider': 1,
            'boundary': 2,
        }

        self.sample_dist=1
        self.num_samples=250
        self.padding = False

        self.ann_file_s3 = ann_file_s3
        self.data_infos_s3 = None
        if self.ann_file_s3 is not None:
            data_infos = mmcv.load(self.ann_file_s3, file_format='pkl')
            if isinstance(data_infos, dict):
                # self.raw_data_keys = list(data_infos.keys())
                data_infos = list(data_infos.values())
            self.data_infos_s3 = data_infos[::self.load_interval]

 

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        meta = {
        'use_lidar': False,
        'use_camera': True,
        'use_external': False,
        'output_format': 'vector',

        # NOTE: please fill the information below
        'method': 'vectormapnet', # name of your method
        'authors': ['foo bar', 'xxx yyy'], # author names
        'e-mail': 'xxx@gmail.com', # your e-mail address
        'institution / company': 'xxx', # your organization
        'country / region': 'XX', # your country/region in iso3166 standard
    }
        data = mmcv.load(ann_file)

        data_infos = []
        for scene_data_key, scene_data in data.items():
            for single_data in scene_data:
                data_infos.append(single_data)

        data_infos = data_infos   # TODO: 注意一下
        
        # data_infos = list(sorted(data_infos, key=lambda e: e['segment_id'] + '_' + e['timestamp']))

        data_infos = data_infos[::self.load_interval]
        # data_infos = data_infos[:200]
        self.metadata = meta        # TODO: 届时需要修改
        # self.metadata = data['metadata']
        # self.version = self.metadata['version']
        return data_infos
    

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)      # NOTE: 这里得到一些 转换矩阵和anno的一些信息
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
       
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue.insert(0, example)

        
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if eval(input_dict['frame_idx']) < eval(frame_idx) and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))        # 如果场景不对的话，那就把原场景copy

        return self.union2one(data_queue)


    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                # prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] = 0
                # metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                # tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] -= prev_pos
                # metas_map[i]['can_bus'][-1] -= prev_angle
                # prev_pos = copy.deepcopy(tmp_pos)
                # prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue
 

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
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
        info = self.data_infos[index]
   
        input_dict = dict(
            sample_idx=(info['segment_id'] + '_' + info['timestamp']),
            pts_filename=None,
            ego2global_translation=np.array(info['pose']['ego2global_translation']),
            ego2global_rotation=np.array(info['pose']['ego2global_rotation']),
            scene_token=info['segment_id'],
            frame_idx=info['timestamp'],
        )

        # ego to global transform
        ego2global = np.eye(4)
        ego2global[:3,:3] = input_dict['ego2global_rotation']
        ego2global[:3, 3] = input_dict['ego2global_translation']
        input_dict['ego2global'] = ego2global 

        image_paths = []
        ego2img_rts = []

        camera_intrinsics = []
        camera_extrinsics = []

        # get intrinsics、extrinsics、 ego2img_rts 
        for cam_type, cam_info in info['sensor'].items():
            if self.ann_file_s3 is None:                # by shengyin
                image_path = cam_info['image_path']
            else:
                image_path = self.data_infos_s3[index]['sensor'][cam_type]['image_path'].replace(
                        f"{self.data_infos_s3[index]['segment_id']}/image", f"{self.data_infos_s3[index]['meta_data']['source_id']}/sensors/cameras", 1)
                
                image_path = os.path.join('s3://odl-flat/Argoverse2/Sensor_Dataset/sensor', image_path)
            
            image_paths.append(image_path)
            cam_extrinsic = np.array(cam_info['extrinsic'])

            # agument for cam_extrinsic 
            if self.noise == 'rotation':
                cam_extrinsic = add_rotation_noise(cam_extrinsic, std=self.noise_std)
            elif self.noise == 'translation':
                cam_extrinsic = add_translation_noise(
                    cam_extrinsic, std=self.noise_std) 

            intrinsic = np.array(cam_info['intrinsic'])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2img_rt = (viewpad @ cam_extrinsic)
            ego2img_rts.append(ego2img_rt)

            camera_intrinsics.append(viewpad)
            camera_extrinsics.append(cam_extrinsic)

        input_dict.update(
            dict(
                img_filenames=image_paths,
                ego2img=ego2img_rts,
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos


        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(np.array(info['pose']['ego2global_rotation']))
        can_bus[:3] = info['pose']['ego2global_translation']
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus

        return input_dict
    
    def get_ann_info(self, index):
        info = self.data_infos[index]
        ann_info = info['annotation']
        
        gt_labels = []
        gt_instance = []

        for k, v in ann_info.items():
            if k in self.cat2id.keys():
                for l in v:
                    gt_labels.append(self.cat2id[k])
                    gt_instance.append(LineString(np.array(l)[:, :2]))      # line with xy

        gt_instance = LiDARInstanceLines(gt_instance,self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size)

        gt_labels = to_tensor(gt_labels)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,

        )
        return anns_results
    

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data



    def draw(self, result, sample_id):
        import cv2
        def _imread( path):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf") # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
            img_bytes = client.get(path)
            assert(img_bytes is not None)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        
        scene_data = self.data_infos[sample_id]
        sample_s3 = self.data_infos_s3[sample_id]

        COLOR_MAP_GT = {  
            'ped_crossing': (255, 0, 0),
            'divider': (0, 0, 255),
            'boundary': (0, 255, 0),
        }

        COLOR_MAP_PRE = {  
            'ped_crossing': (155, 0, 0),
            'divider': (0, 0, 155),
            'boundary': (0, 155, 0),
        }

        map_size=[-55, 55, -30, 30]
        scale=10

        draw_gt_dir = 'draw_gt'
        path_dataroot = 's3://odl-flat/Argoverse2/Sensor_Dataset/sensor'
        
        for frame in scene_data:
            sensor = scene_data['sensor']

            for cam in sensor.keys():
                path_img = self.data_infos_s3[sample_id]['sensor'][cam]['image_path'].replace(
                        f"{self.data_infos_s3[sample_id]['segment_id']}/image", f"{self.data_infos_s3[sample_id]['meta_data']['source_id']}/sensors/cameras", 1)
                
                path_img = os.path.join(path_dataroot, path_img)
                img = _imread(path_img)

                intrinsic = np.array(scene_data['sensor'][cam]['intrinsic'])
                extrinsic = np.array(scene_data['sensor'][cam]['extrinsic']) 

                viewpad = np.eye(4)     
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2img = (viewpad @ extrinsic)


                # for bev
                bev_image_gt = np.zeros((int(scale*(map_size[1]-map_size[0])), int(scale*(map_size[3] - map_size[2])), 3), dtype=np.uint8)
                bev_image_pred = np.zeros((int(scale*(map_size[1]-map_size[0])), int(scale*(map_size[3] - map_size[2])), 3), dtype=np.uint8)

                for name, pts in scene_data['annotation'].items():

                    # for img
                    for line in pts:
                        line = np.array(line)[:, :3]
                        line[:, 2] = 0.0        # 这里强行赋值 z为0 
                        xyz1 = np.concatenate([line, np.ones((line.shape[0], 1))], axis=1)
                        xyz1 = xyz1 @ ego2img.T
                        xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                        if xyz1.shape[0] == 0:
                            continue
                        points_2d = xyz1[:, :2] / xyz1[:, 2:3]
                        # mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])
                        points_2d = points_2d.astype(int)
                            
                        img = cv2.polylines(img, points_2d[None], False, COLOR_MAP_GT[name], 2)

                    for vec, score, label in zip(result['vectors'], result['scores'], result['labels']):
                        if score < 0.1:
                            continue
                        vec = np.concatenate([vec, np.zeros((vec.shape[0], 1))], axis=1)
                        xyz1 = np.concatenate([vec, np.ones((vec.shape[0], 1))], axis=1)
                        xyz1 = xyz1 @ ego2img.T
                        xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                        if xyz1.shape[0] == 0:
                            continue
                        points_2d = xyz1[:, :2] / xyz1[:, 2:3]
                        # mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])
                        points_2d = points_2d.astype(int)
                            
                        img = cv2.polylines(img, points_2d[None], False, COLOR_MAP_PRE[list(COLOR_MAP_PRE.keys())[label]], 2) 


            
                    for lane in pts:
                        draw_coor = (scale * (-np.array(lane)[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                        bev_image_gt = cv2.polylines(bev_image_gt, [draw_coor[:, [1,0]]], False, COLOR_MAP_GT[name], max(round(scale * 0.2), 1))
                        bev_image_gt = cv2.circle(bev_image_gt, (draw_coor[0, 1], draw_coor[0, 0]), max(2, round(scale * 0.5)), COLOR_MAP_GT[name], -1)
                        bev_image_gt = cv2.circle(bev_image_gt, (draw_coor[-1, 1], draw_coor[-1, 0]), max(2, round(scale * 0.5)) , COLOR_MAP_GT[name], -1)


                    for lane, score, label in zip(result['vectors'], result['scores'], result['labels']):
                        if score < 0.1:
                            continue
                        draw_coor = (scale * (-np.array(lane)[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                        bev_image_pred = cv2.polylines(bev_image_pred, [draw_coor[:, [1,0]]], False, COLOR_MAP_PRE[list(COLOR_MAP_PRE.keys())[label]], max(round(scale * 0.2), 1))
                        bev_image_pred = cv2.circle(bev_image_pred, (draw_coor[0, 1], draw_coor[0, 0]), max(2, round(scale * 0.5)), COLOR_MAP_PRE[list(COLOR_MAP_PRE.keys())[label]], -1)
                        bev_image_pred = cv2.circle(bev_image_pred, (draw_coor[-1, 1], draw_coor[-1, 0]), max(2, round(scale * 0.5)) , COLOR_MAP_PRE[list(COLOR_MAP_PRE.keys())[label]], -1)

                    

                mmcv.mkdir_or_exist(os.path.join(f'{draw_gt_dir}', f'{scene_data["segment_id"]}', f'{scene_data["timestamp"]}'))
                cv2.imwrite(os.path.join(f'{draw_gt_dir}', f'{scene_data["segment_id"]}', f'{scene_data["timestamp"]}', f'{cam}.png'), img)
                
                bev_image_overlab = bev_image_gt + bev_image_pred
                bev_image = np.concatenate([bev_image_pred, bev_image_overlab, bev_image_gt], axis=1)
                cv2.imwrite(os.path.join(f'{draw_gt_dir}', f'{scene_data["segment_id"]}', f'{scene_data["timestamp"]}', f'bev.png'), bev_image)
                



    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        assert self.map_ann_file is not None
        pred_annos = {}
        mapped_class_names = self.MAPCLASSES
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            vecs = output_to_vecs(det)

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            for i, vec in enumerate(vecs):
                # name = mapped_class_names[vec['label']]
                
                single_case['vectors'].append(vec['pts'])
                single_case['scores'].append(vec['score'])
                single_case['labels'].append(vec['label'])
       
            sample_token = self.data_infos[sample_id]['timestamp']
            pred_annos[sample_token] = single_case
            # self.draw(single_case, sample_id)

        nusc_submissions = {
            'meta': self.modality,
            'results': pred_annos,

        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'nuscmap_results.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def to_gt_vectors(self,
                      gt_dict):
        # import pdb;pdb.set_trace()
        gt_labels = gt_dict['gt_labels_3d'].data
        gt_instances = gt_dict['gt_bboxes_3d'].data.instance_list

        gt_vectors = []

        for gt_instance, gt_label in zip(gt_instances, gt_labels):
            pts, pts_num = sample_pts_from_line(gt_instance, patch_size=self.patch_size)
            gt_vectors.append({
                'pts': pts,
                'pts_num': pts_num,
                'type': int(gt_label)
            })
        vector_num_list = {}
        for i in range(self.NUM_MAPCLASSES):
            vector_num_list[i] = []
        for vec in gt_vectors:
            if vector['pts_num'] >= 2:
                vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))
        return gt_vectors


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
        self.evaluator = VectorEvaluate(self.ann_file)
        
        results_dict = self.evaluator.evaluate(result_files['pts_bbox'], logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict


def output_to_vecs(detection):
    box3d = detection['boxes_3d'].numpy()
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    pts = detection['pts_3d'].numpy()

    vec_list = []
    for i in range(box3d.shape[0]):
        vec = dict(
            bbox = box3d[i], # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list

def sample_pts_from_line(line, 
                         fixed_num=-1,
                         sample_dist=1,
                         normalize=False,
                         patch_size=None,
                         padding=False,
                         num_samples=250,):
    if fixed_num < 0:
        distances = np.arange(0, line.length, sample_dist)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    else:
        # fixed number of points, so distance is line.length / fixed_num
        distances = np.linspace(0, line.length, fixed_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

    if normalize:
        sampled_points = sampled_points / np.array([patch_size[1], patch_size[0]])

    num_valid = len(sampled_points)

    if not padding or fixed_num > 0:
        # fixed num sample can return now!
        return sampled_points, num_valid

    # fixed distance sampling need padding!
    num_valid = len(sampled_points)

    if fixed_num < 0:
        if num_valid < num_samples:
            padding = np.zeros((num_samples - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:num_samples, :]
            num_valid = num_samples

        if normalize:
            sampled_points = sampled_points / np.array([patch_size[1], patch_size[0]])
            num_valid = len(sampled_points)

    return sampled_points, num_valid