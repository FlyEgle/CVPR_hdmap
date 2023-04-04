'''
Descripttion: UV images segmentation heads & loss function
Author: jiangmingchao
version: 1.0
Date: 2023-02-13 07:13:07
'''
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List, Optional, Union, Dict, Any
from mmcv.runner import auto_fp16, force_fp32
from torch.cuda.amp import autocast as autocast
from mmcv.cnn import build_conv_layer
import numpy as np
import math
import copy
from mmdet.models import HEADS

# __all__ = ["uvSegmentationsAuxHead", "AuxBEVSegmentationHead"]


def sigmoid_xent_loss(inputs: torch.Tensor, targets: torch.Tensor,reduction: str = "mean") -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:    
    inputs = inputs.float()
    targets = targets.float()

    num = inputs.size(0)
    smooth = 1e-3

    probs = torch.sigmoid(inputs)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = m1 * m2 
    
    score = 2 * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num 
    return score 

class upConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upConvModule, self).__init__()
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    # @auto_fp16()
    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )
        out = self.convbnrelu(x)
        return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    # @auto_fp16()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(SELayer, self).__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()
    # @auto_fp16()
    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class uvSegmentationsAux_layer(nn.Module):
    def __init__(self,in_channels, mid_channels):
        super(uvSegmentationsAux_layer, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.bn = nn.BatchNorm1d(4*4)        # for cam-aware
        self.context_mlp = Mlp(4*4, mid_channels, mid_channels)     # TODO: 这里最好还是需要改成 4*4*7
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.context_conv = nn.Sequential(*[
            nn.Conv2d(
                mid_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            # build_conv_layer(         # NOTE: 由于 mmcv 版本的问题，这里DCN还不支持
            #         cfg=dict(
            #             type='DCN',
            #             in_channels=in_channels,
            #             out_channels=in_channels,
            #             kernel_size=3,
            #             padding=1,
            #             groups=4,
            #             im2col_step=128,
            #         )),
        ])

    # @auto_fp16()
    def forward(self, x, mlp_input):
        import pdb; pdb.set_trace()
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1] *mlp_input.shape[-1]))
        # import pdb; pdb.set_trace()        
        # x = x.to(torch.FloatTensor)
  
        x = self.reduce_conv(x)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
     
        return context

@HEADS.register_module()
class uvSegmentationsAuxHead(nn.Module):
    """uv images segmentations with aux loss
    """
    def __init__(self, 
                 target_size=[256, 704], 
                 in_channels=256, 
                 out_channels=64, 
                 scales=[8, 16, 32],
                 loss_segmentations_type='xent_dice',
                 camera_type = ['ring_front_center', 'ring_front_left', 'ring_front_right', \
                                'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right'],
 
                 ):
        super(uvSegmentationsAuxHead, self).__init__()
        
        self.loss_segmentations_type = loss_segmentations_type 
        self.loss_segmentations = self._xent_dice_loss if loss_segmentations_type=='xent_dice' else self._xent_loss
        self.camera_type = camera_type

        self.target_size = target_size

        self.in_channels = in_channels
        scales = sorted(scales, reverse=True)
        self.out_channels = [
         list(
            np.linspace(in_channels, out_channels, int(math.log(scales[i], 2))).astype(int)
            )  
                for i in range(len(scales))
        ]   

        # self.target_size = [256, 704]
        # self.in_channels = 256
        # self.out_channels = [
        #     [256, 224, 192, 160, 64],  # stage 1
        #     [256, 192, 160, 64],  # stage 2
        #     [256, 192, 64]   # stage 3 
        # ]

        # stage
        self.predict_head = [[] for _ in range(len(self.out_channels))]

        self.strengModule = []
        self.headModule = nn.ModuleList()
        for i in range(len(self.out_channels)):
            for l in range(len(self.out_channels[i])):
                
                if l == 0:
                    self.strengModule.append(uvSegmentationsAux_layer(self.in_channels, self.in_channels//2))
                    up_convbnrelu = upConvModule(self.in_channels, self.out_channels[i][l])
                else:
                    up_convbnrelu = upConvModule(self.out_channels[i][l-1], self.out_channels[i][l])
                
                self.predict_head[i].append(up_convbnrelu)
            
            # module sequence
            self.headModule.append(nn.Sequential(*self.predict_head[i]))

        # output 
        # self.out = nn.ModuleList([
        #     nn.Conv2d(self.out_channels[0][-1], 1, 3, 1, 1),
        #     nn.Conv2d(self.out_channels[1][-1], 1, 3, 1, 1),
        #     nn.Conv2d(self.out_channels[2][-1], 1, 3, 1, 1)
        #     ]
        # )

        # self.outconv = nn.Conv2d(3, 1, 1)

        self.out = nn.ModuleList([
        nn.Conv2d(self.out_channels[i][-1], 1, 3, 1, 1) 
                for i in range(len(scales))
            ]
        )

        self.outconv = nn.Conv2d(len(scales), 1, 1)
    

    def _xent_loss(self, x, target):
        return sigmoid_xent_loss(x, target)

    def _xent_dice_loss(self, x, target):
        return sigmoid_xent_loss(x, target) + dice_loss(x, target) 


    # @auto_fp16()
    def forward(self, inputs, lidar2imgs=None):
        assert isinstance(inputs, List) or isinstance(inputs, tuple)
        out_list = []
        for i in range(len(inputs)):
            # stage
            
            # input = self.strengModule[i](inputs[i], lidar2imgs)
            input = inputs[i]
            input = input.flatten(0, 1)
            stage_out = self.headModule[i](input)
            # out
            out = self.out[i](stage_out)
            out_list.append(out)
        
        # ((B*T), C, H, W)
      
        outcat = torch.cat(out_list, dim=1)
        out = self.outconv(outcat)

        return out 
    
    def get_segmentation_loss(self, x, target):
        if isinstance(x, (list, tuple)):
            x = x[0]
        # (b*t) * c * h * w -> b * t * c * h * w
        bt, c, h, w = x.shape 
        x = x.view(bt // len(self.camera_type), len(self.camera_type) * c, h, w)
        target = target/target.max()
        # print("view x shape: ", x.shape)
        # print("target shape: ", target.shape)
        losses = {}

        for index, name in enumerate(self.camera_type):
            loss = self.loss_segmentations(x[:, index], target[:, index])
            # if self.loss == "xent":
            #     loss = sigmoid_xent_loss(x[:, index], target[:, index])
            # elif self.loss == "xent_dice":
            #     bceloss = sigmoid_xent_loss(x[:, index], target[:, index])
            #     diceloss = dice_loss(x[:, index], target[:, index])
            #     loss = bceloss + diceloss
            # else:
            #     raise ValueError(f"unsupported loss: {self.loss}")
            losses[f"{name}/{self.loss_segmentations_type}"] = loss
            import pdb; pdb.set_trace() 
        return losses
       


class AuxBEVSegmentationHead(nn.Module):
    """aux bev segmentation head 
    """
    def __init__(
        self,
        classes: List[str],
        loss: str,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.loss = loss
        self.t = 6  # T frames
        self.frame_names = ["front_60", "front_right", "back_right", "back", "back_left", "front_left"]

    # @auto_fp16()
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,   # (b, t, h, w)
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]
        # (b*t) * c * h * w -> b * t * c * h * w
        bt, c, h, w = x.shape 
        x = x.view(bt // self.t, self.t * c, h, w)
        # print("view x shape: ", x.shape)
        # print("target shape: ", target.shape)
        if self.training:
            losses = {}
            for index, name in enumerate(self.frame_names):
                if self.loss == "xent":
                    loss = sigmoid_xent_loss(x[:, index], target[:, index])
                elif self.loss == "xent_dice":
                    bceloss = sigmoid_xent_loss(x[:, index], target[:, index])
                    diceloss = dice_loss(x[:, index], target[:, index])
                    loss = bceloss + diceloss
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}/{self.loss}"] = loss
                
            return losses
        else:
            return torch.sigmoid(x)





if __name__ == "__main__":
    model = uvSegmentationsAuxHead()
    # model = upConvModule(3,3)
    model = model
    # print(model)
    inputs = [
        torch.randn(7, 256, 8, 22),
        torch.randn(7, 256, 16, 44),
        torch.randn(7, 256, 32, 88)
    ]
    # inputs = torch.randn(1,3,32,32).cuda()
    # with autocast():
    lidar2imgs = torch.zeros((7, 4, 4))
    out = model(inputs, lidar2imgs)
    print(inputs[0].dtype)
    print(out.dtype)
    
