import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
import math
@TRANSFORMER.register_module()
class MapTRPerceptionTransformerTWOSTAGE(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                #  num_cams=6, use for nuscene
                num_cams=7, # use for av2
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],

                 bev_seg_head=None,
                 **kwargs):
        super(MapTRPerceptionTransformerTWOSTAGE, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

        self.bev_seg_head = bev_seg_head
        if self.bev_seg_head is not None:
            try:
                from mmseg.models import builder
                self.bev_seg_head = builder.build_head(bev_seg_head)
            except:
                from mmdet3d.models import builder
                self.bev_seg_head = builder.build_head(bev_seg_head)
        
        self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)

        self.pos_embedding = nn.Linear(2, self.embed_dims)

    def _for_init(self, ):
        mask = torch.zeros((2, 2, 100, 200))

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) # TODO, this is a hack
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed
    

    @staticmethod
    def get_bev_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(          # TODO: 需要check 0.5 作用
                torch.linspace(
                    0.0, H - 1.0, H, dtype=dtype, device=device),
                torch.linspace(
                    0.0, W - 1.0, W, dtype=dtype, device=device)
            )
            ref_y = ref_y[None] / H
            ref_x = ref_x[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)            # TODO: 这里 y x 的顺序
            ref_2d = ref_2d.repeat(bs, 1, 1, 1)
            return ref_2d


    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        
        B, _, C = bev_embed.shape
        bev_feature = bev_embed.permute(1,2, 0).reshape(B, C, bev_h, bev_w)
        bev_seg = self.bev_seg_head([bev_feature])


        # # ==== 下面是二阶段的
        # bev_seg_prob = torch.softmax(bev_seg, dim=1)
        # foreground_bev_seg = bev_seg_prob[:, 1:]     # 获取到前景

        # local_bev_seg = get_local_maximum(foreground_bev_seg)
        # ref_2d = self.get_bev_points(
        #     bev_h, bev_w, dim='2d', bs=B, device=bev_embed.device, dtype=bev_embed.dtype)

        # value, index = local_bev_seg.view(B,-1).topk(50*20,dim=1)
        # reference_points = torch.gather(ref_2d.view(B,-1,2), 1, index.unsqueeze(-1).repeat(1,1,2))
        # # reference_points = reference_points.detach()
        # # import pdb; pdb.set_trace()
        # pos_trans_out = self.pos_trans_fc(
        #         self.get_proposal_pos_embed(reference_points, num_pos_feats=self.embed_dims))
        # pos_trans_out = self.pos_trans_norm(pos_trans_out)
        # query_pos, query = torch.split(pos_trans_out, C, dim=2)
        # # =====

        # ==== 下面是二阶段的 实验二
        bev_seg_prob = torch.softmax(bev_seg, dim=1)
        foreground_bev_seg = bev_seg_prob[:, 1:]     # 获取到前景
        local_bev_seg = get_local_maximum(foreground_bev_seg)
        ref_2d = self.get_bev_points(
            bev_h, bev_w, dim='2d', bs=B, device=bev_embed.device, dtype=bev_embed.dtype)
        value, index = local_bev_seg.view(B,-1).topk(50*20,dim=1)

        reference_points = torch.gather(ref_2d.view(B,-1,2), 1, index.unsqueeze(-1).repeat(1,1,2))
        query = object_query_embed
        query = query.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.pos_embedding(reference_points)
        # =====


        # ===
        # cv2.imwrite('a.png', (foreground_bev_seg[0,0].cpu().detach().numpy()*255))
        # cv2.imwrite('a1.png', (local_bev_seg[0,0].cpu().detach().numpy()*255))
        # cv2.imwrite('a2.png', (foreground_bev_seg[0,0].cpu().detach().numpy()*255))
       
        # === 下面是只训练seg
        # bev_embed = None
        # inter_states = None
        # init_reference_out = None
        # inter_references_out = None
        
        # =============


        # # ==== 下面是一阶段的
        # bs = mlvl_feats[0].size(0)
        # query_pos, query = torch.split(
        #     object_query_embed, self.embed_dims, dim=1)
        # query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        # query = query.unsqueeze(0).expand(bs, -1, -1)
        # reference_points = self.reference_points(query_pos)
        # reference_points = reference_points.sigmoid()
        # # ====
        
        init_reference_out = reference_points               
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out, bev_seg
    
    @staticmethod
    def get_proposal_pos_embed(proposals,
                               num_pos_feats=256,
                               temperature=10000):
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        proposals = proposals * scale
        # N, L, 2, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos



def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep