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


def intrinsic_project(resize_ratio, cam_intrinsic):
    """image->resize : intrinsic->resize
    Args:
        resize_ratio: [w_ratio, h_ratio]
        cam_intrinsic: np.ndarray [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Returns:
        cam_intrinsic
    """
    cam_intrinsic[0, :] = cam_intrinsic[0, :] * resize_ratio[0]
    cam_intrinsic[1, :] = cam_intrinsic[1, :] * resize_ratio[1]
    return cam_intrinsic 


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
        self.max_x = self.patch_size[1] / 2   # 30
        self.max_y = self.patch_size[0] / 2   # 15
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
        instance_se_points_tensor[:,0] = torch.clamp(instance_se_points_tensor[:,0], min=-self.max_x,max=self.max_x)  # (-30, 30)
        instance_se_points_tensor[:,1] = torch.clamp(instance_se_points_tensor[:,1], min=-self.max_y,max=self.max_y)  # (-15, 15)
        instance_se_points_tensor[:,2] = torch.clamp(instance_se_points_tensor[:,2], min=-self.max_x,max=self.max_x)  # (-30, 30)
        instance_se_points_tensor[:,3] = torch.clamp(instance_se_points_tensor[:,3], min=-self.max_y,max=self.max_y)  # (-15, 15)
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
                # import pdb;pdb.set_trace()
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


# vectorized the local map 
class VectorizedLocalMap(object):
    # CLASS2LABEL = {
    #     'divider'     : 0,
    #     'road_divider': 0,
    #     'lane_divider': 0,
    #     'ped_crossing': 1,
    #     'boundary'    : 2,
    #     'contours': 2,
    #     'others': -1
    # }
    CLASS2LABEL = {
            'ped_crossing': 0,
            'divider': 1,
            'boundary': 2,
    }
    def __init__(self,
                 dataroot,
                 patch_size,
                 map_classes=['ped_crossing','divider','boundary'],
                #  line_classes=['road_divider', 'lane_divider'],
                line_classes = ['divider', 'boundary'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000,):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        # self.data_root = dataroot
        # self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
        #              'singapore-onenorth', 'singapore-queenstown']
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        # self.nusc_maps = {}
        # self.map_explorer = {}
        # for loc in self.MAPS:
        #     self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
        #     self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        # print(self.nusc_maps)  
        # import pdb;pdb.set_trace() 

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value


    # main interface
    def gen_vectorized_samples(self, input_dict):
        '''vectorized the ped_crossing, divider & boundary follow the MapTR format
        # V1.0 divier, boundary is belong to the LineString, ped_crossing is belong to Polygon
        Args:
            
            input_dict : "ped_crossing": list [        -- list of ped crossing lines
                            line1 (N1 x 4),            -- line as list of points, each point as (x, y, z, visibility)
                                                          visibility: 1 for visible at current frame, 0 for occluded
                            line2 (N2 x 4),
                            ...
                        ]
                        "divider": list [...],         -- list of divider lines
                        "boundary": list [...]         -- list of boundary lines
        
        Returns:
            
            anns_result : {
                "gt_vecs_pts_loc",
                "gt_vecs_label",
                "gt_anns" : lidar_anno
            }
        '''
        vectors = []
        anns = input_dict["lidar_anno"]
        # for vec_class in self.vec_classes:
        line_geom = self.get_map_geom_from_lidar(anns, self.vec_classes)
        line_instances_dict = self.line_geoms_to_instances(line_geom)  
        # if vec_class == "divider" or vec_class == "boundary":   
   
        for line_type, instances in line_instances_dict.items():
            for instance in instances:
                vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            # elif vec_class == 'ped_crossing':
            #     # print("ped_crossing: ", anns[vec_class])
            #     if anns[vec_class] != []:
            #         ped_geom = self.get_map_geom_from_lidar(anns, self.ped_crossing_classes)
            #         # print("ped_geom: ", ped_geom)
            #         ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
            #         for instance in ped_instance_list:
            #             vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            # else:
            #     raise ValueError(f'WRONG vec_class: {vec_class}')

        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)
        
        gt_instance = LiDARInstanceLines(gt_instance, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
            gt_anns = anns

        )
        return anns_results

    def get_linestring_from_lidar(self, anns, layer_name):
        """get List[LineString] for divider & boundary
        """
        if layer_name not in anns:
            raise ValueError("{} is not a line layer".format(layer_name))
        
        # mutl line for divider
        lane_list = []
        for lane in anns[layer_name]:
            new_points = np.array(lane)[:, :2]  # x,y,z,v->x,y
            lane_list.append(LineString(new_points))
        # import pdb;pdb.set_trace()

        return lane_list 

    def get_polygon_from_lidar(self, anns, layer_name):
        """get List[Polygon] for ped_crossing
        """
        if layer_name not in anns:
            raise ValueError("{} is not a line layer".format(layer_name))
        
        polygon_list = []
        for polygon in anns[layer_name]:    
            new_points = np.array(polygon)[:, :2] # x,y,z,v->x,y
            polygon_list.append(Polygon(new_points))

        return polygon_list

    def get_map_geom_from_lidar(self, anns, layer_names):
        """Returns map geom include LineString or Polygon
        """
        map_geom = []
        for layer_name in layer_names:
            
            # if layer_name in self.ped_crossing_classes: # ['ped_crossing']
            #     geoms = self.get_polygon_from_lidar(anns, layer_name)
            #     # print("geoms: ", geoms)
            #     map_geom.append([layer_name, geoms])

            if layer_name in self.vec_classes:  # ['divider', 'boundary', 'ped_crossing']
                geoms = self.get_linestring_from_lidar(anns, layer_name)
                map_geom.append([layer_name, geoms])

        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        # for p in ped:
        #     print("-------------------------pedcrossing-----------------------")
        #     for coord in p.exterior.coords:
        #         print("coord: ", c)
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict
    
    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)


        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples


        return sampled_points, num_valid

# ====================================== main dataset ====================================
# av2 dataset
@DATASETS.register_module()
# class CustomAV2MapDataset(CustomNuScenesDataset):
class CustomAV2MapDataset(Dataset):
    r"""AV2 Map dataset with lidar annotations

    This dataset add static map elements
    """
    def __init__(self, 
                #  data_root,
                 map_ann_file = None,
                 out_ann_file = None,
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
                 test_mode=False,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 
                 ann_file_s3=None,
                 data_infos_s3=None, 
                 *args, 
                 **kwargs):
        # super().__init__(*args, **kwargs)
        # CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        #        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
        #        'barrier')
        # CLASSES = ('divider', 'boundary', 'ped_crossing')

        super().__init__()
        self.map_ann_file = map_ann_file 
        self.out_ann_file = out_ann_file

        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.data_infos = self._load_annotations(self.map_ann_file) # debug
        self.CLASSES = ('divider', 'ped_crossing', 'boundary')
        # self.CLASSES = ('ped_crossing', 'divider', 'boundary')
        self.MAPCLASSES = self.get_map_classes(map_classes)  # ['divider', 'ped_crossing','boundary']
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w) # h, w ->nuscense is 60, 30, av2 is 30, 60
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        
        self.modality = modality
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        # load anno file
        print(self.map_ann_file)
        
        self.flag = np.zeros(len(self), dtype=np.uint8)

        # vectorized the location map for nuscense
        self.vector_map = VectorizedLocalMap(kwargs["data_root"], 
                            patch_size=self.patch_size, map_classes=self.MAPCLASSES, 
                            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
                            padding_value=self.padding_value)
        
        self.is_vis_on_test = False
        self.noise = noise
        self.noise_std = noise_std
        self.ann_file_s3 = ann_file_s3
        self.data_infos_s3 = None
        if self.ann_file_s3 is not None:
            data_infos = mmcv.load(self.ann_file_s3, file_format='pkl')
            if isinstance(data_infos, dict):
                # self.raw_data_keys = list(data_infos.keys())
                data_infos = list(data_infos.values())
            self.data_infos_s3 = data_infos
        
    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names

    def vectormap_pipeline(self, example, input_dict):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(matrix=np.array(input_dict['lidar2ego_rotation'])).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(matrix=np.array(input_dict['ego2global_rotation'])).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        # location = input_dict['map_location']
        # import pdb; pdb.set_trace()
        ego2global_translation = input_dict['ego2global_translation']
        ego2global_rotation = input_dict['ego2global_rotation']


        # vectorized samples
        anns_results = self.vector_map.gen_vectorized_samples(input_dict)
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index,
            'gt_anns' : lidar_anns
        '''

        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)

        # example.update(dict(gt_anns=anns_results['gt_anns']))
        return example

    def _load_annotations(self, ann_file):
        # return super().load_annotations(ann_file)
        """Load annotations form ann_file.

        Args:
            ann_file (str): Path of the annotation file.
        
        Returns:
            list[dict] : List of each samples annotation

        """
        self.ann_file = ann_file
        ann = mmcv.load(ann_file)
        samples = []
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
        # samples_list = samples_list[:100:4]
        return samples_list
        
    def get_ann_info(self, index):
        # return super().get_ann_info(index)
        """Get annotation info according to the given index.
        Because the av2 have no 3d movable object, return all is none or []

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # info = self.data_infos[index]
        gt_bboxes_3d = []
        gt_names_3d  = []
        gt_labels_3d = []

        anno_info = self.data_infos[index]['annotation']

        gt_instance_xyz_list = []
        for k, v in anno_info.items():
            for l in v:
                gt_instance_xyz_list.extend([np.array(l)[:, :3]])

        anns_results = dict(
            gt_bboxes_3d = gt_bboxes_3d,
            gt_labels_3d = gt_labels_3d,
            gt_names = gt_names_3d,
            gt_instance_xyz_list=gt_instance_xyz_list,
        )

        return anns_results


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

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        # pre process the input to a default format
        self.pre_pipeline(input_dict)
        # TODO: dataaug for 3d bbox, need to modify
        example = self.pipeline(input_dict)
        # important function to vectorlize the map 
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None

            
        # # # 可视化 gt
        # import cv2
        # import copy
        # color_type = {
        #     0: (0, 0, 255),
        #     1: (255, 0, 0),
        #     2: (0, 255, 0),
        # }
        # for index, img in enumerate(example['img']._data):
        #     img = copy.deepcopy(img)
        #     img = torch.permute(img, (1, 2, 0))
        #     img = np.uint8(img)
        #     img = np.ascontiguousarray(img, dtype=np.uint8)

        #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     lidar2img = example['img_metas']._data['lidar2img'][index]
        #     for i, instance in enumerate(example['gt_bboxes_3d']._data.instance_list):
        #         instance = np.array(instance.coords)
        #         instance = np.concatenate([instance, np.zeros((instance.shape[0], 1))], axis=1)
        #         instance = np.concatenate([instance, np.ones((instance.shape[0], 1))], axis=1)
        #         instance = instance @ lidar2img.T
        #         instance = instance[instance[:, 2] > 1e-5]
        #         # if xyz1.shape[0] == 0:
        #         #     continue
        #         points_2d = instance[:, :2] / instance[:, 2:3]
        #         # mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])
        #         points_2d = points_2d.astype(int)
                
        #         img = cv2.polylines(img, points_2d[None], False, color_type[example['gt_labels_3d']._data.numpy()[i]], 2)

        #     CAM_TYPE = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']
        #     dir = f"../instance_draw/{input_dict['scene_token']}/{input_dict['sample_idx']}"
        #     mmcv.mkdir_or_exist(dir)
        #     import os
        #     cv2.imwrite(os.path.join(dir, f"{CAM_TYPE[index]}.png" ), img)

          

        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = self.vectormap_pipeline(example,input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
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
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue


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
                folder = self.map_ann_file.split("/")[-1].split("_")[0]
                prefix = self.map_ann_file.split("/OpenLaneV2")[0] + "/OpenLaneV2"  # hard code
                # prefix = self.map_ann_file.split("/")[-1].split("_")[0]
                prefix_path = os.path.join(prefix, folder)
                if self.ann_file_s3 is None:                # by shengyin
                    image_path = os.path.join(prefix_path, cam_info["image_path"])
                else:
                    image_path = self.data_infos_s3[index]['sensor'][cam_type]['image_path'].replace(
                        f"{self.data_infos_s3[index]['segment_id']}/image", f"{self.data_infos_s3[index]['meta_data']['source_id']}/sensors/cameras", 1)
                
                    image_path = os.path.join('s3://odl-flat/Argoverse2/Sensor_Dataset/sensor', image_path)
                image_paths.append(image_path)

                # TODO: this is hard code, need modify 
                # mmcv.imread -> (h, w, c)
                
                if self.ann_file_s3 is not None:
                    img_height, img_width, _ = self._imread(image_path).shape

                else:
                    img_height, img_width, _ = mmcv.imread(os.path.join(prefix_path,
                                                                    cam_info["image_path"])).shape 
                # constant_resize_shape = (1600, 900)
                constant_resize_shape = (2048, 1550)

                resize_ratio = [constant_resize_shape[0] / img_width, 
                constant_resize_shape[1] / img_height]

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
                intrinsic = intrinsic_project(resize_ratio, intrinsic) # TODO: hard code
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

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

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
        if self.is_vis_on_test:
            example = self.vectormap_pipeline(example, input_dict)
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

    def __len__(self):
        return len(self.data_infos)

    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        # assert self.map_ann_file is not None
        self.out_ann_file = './re.pkl'
        assert self.out_ann_file is not None
        print("out_ann_file: ", self.out_ann_file)
        if (not os.path.exists(self.out_ann_file)) :
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['timestamp']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                input_dict = self.get_data_info(sample_id)
                # gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, input_dict)
                gt_labels = gt_sample_dict['gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['gt_bboxes_3d'].data.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.out_ann_file)
            mmcv.dump(nusc_submissions, self.out_ann_file)
        else:
            print(f'{self.out_ann_file} exist, not update')

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir


    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        detail = dict()
        
        print('Formating results & gts by classes')
        with open(result_path,'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        
        with open(self.out_ann_file,'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']

        # print("annotations: ", annotations)
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)
        # print("cls_gens: ", cls_gens)
        # print("cls_gts: ", cls_gts)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)

            if metric == 'chamfer':
                thresholds = [0.5,1.0,1.5]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=logger,
                                num_pred_pts_per_instance=self.fixed_num,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail


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
        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # if isinstance(result_files, dict):
        #     results_dict = dict()
        #     for name in result_names:
        #         print('Evaluating bboxes of {}'.format(name))
        #         ret_dict = self._evaluate_single(result_files[name], metric=metric)
        #     results_dict.update(ret_dict)
        # elif isinstance(result_files, str):
        #     results_dict = self._evaluate_single(result_files, metric=metric)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        self.evaluator = VectorEvaluate(self.ann_file)
        submisson_vector_path = result_files['pts_bbox']
        submisson_vector = mmcv.load(submisson_vector_path)
        submisson_vector['meta'] = {
                'use_lidar': False,
                'use_camera': True,
                "use_external": False,                     
                "output_format": "vector",                  
                'use_external': False,

                # NOTE: please fill the information below
                'method': 'maptr',                            
                'authors': ['JiangShengyin'],                          
                'e-mail': 'shengyin@bupt.edu.cn',                            
                'institution / company': 'bupt',         
                'country / region': 'china',                  
        }
        mmcv.dump(submisson_vector, 'submisson_vector.json')
        results_dict = self.evaluator.evaluate(result_files['pts_bbox'], logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict




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
                # if vec['label'] == 0:
                #     vec['label'] = 1
                # elif vec['label'] == 1:
                #     vec['label'] = 0
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
            'ped_crossing': (255, 255, 255),
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
                




    # def _format_bbox(self, results, jsonfile_prefix=None):
    #     """Convert the results to the standard format.

    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         jsonfile_prefix (str): The prefix of the output jsonfile.
    #             You can specify the output directory/filename by
    #             modifying the jsonfile_prefix. Default: None.

    #     Returns:
    #         str: Path of the output json file.
    #     """
    #     assert self.map_ann_file is not None
    #     pred_annos = []
    #     mapped_class_names = self.MAPCLASSES
    #     # import pdb;pdb.set_trace()
    #     print('Start to convert map detection format...')
    #     for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
    #         pred_anno = {}
    #         vecs = output_to_vecs(det)
    #         sample_token = self.data_infos[sample_id]['timestamp']
    #         pred_anno['sample_token'] = sample_token
    #         pred_vec_list=[]
    #         for i, vec in enumerate(vecs):
    #             name = mapped_class_names[vec['label']]
    #             anno = dict(
    #                 pts=vec['pts'],
    #                 pts_num=len(vec['pts']),
    #                 cls_name=name,
    #                 type=vec['label'],
    #                 confidence_level=vec['score'])
    #             pred_vec_list.append(anno)

    #         pred_anno['vectors'] = pred_vec_list
    #         pred_annos.append(pred_anno)

    #     self._format_gt()
    #     # if not os.path.exists(self.out_ann_file):
    #     #     self._format_gt()
    #     # else:
    #     #     print(f'{self.out_ann_file} exist, not update')

    #     nusc_submissions = {
    #         'meta': self.modality,
    #         'results': pred_annos,

    #     }

    #     mmcv.mkdir_or_exist(jsonfile_prefix)
    #     res_path = osp.join(jsonfile_prefix, 'nuscmap_results.json')
    #     print('Results writes to', res_path)
    #     mmcv.dump(nusc_submissions, res_path)
    #     return res_path

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
