import math

import torch
import torch.nn as nn
from nnunetv2.training.models.model_builder import ModelBuilder
from nnunetv2.utilities.find_class_by_name import my_import
from nnunetv2.training.models.constant import CONCAT, ADD, TWO_D, THREE_D

import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import torch
import torch.nn as nn
#CAM start=================================

import torch
import torch.nn as nn
from einops.layers.torch import Reduce
from nnunetv2.inference.mc_dropout.mcdropout import *

class ModuleStateController:
    TWO_D = "2d"
    THREE_D = "3d"

    state = TWO_D
    def __init__(self):
        assert False, "Don't make this object......"
    
    @classmethod
    def conv_op(clss):
        if clss.state == clss.THREE_D:
            return nn.Conv3d
        else:
            return nn.Conv2d
        
    @classmethod
    def norm_op(clss):
        if clss.state == clss.THREE_D:
            return nn.InstanceNorm3d
        else:
            return nn.InstanceNorm2d

    @classmethod
    def transp_op(clss):
        if clss.state == clss.THREE_D:
            return nn.ConvTranspose3d
        else:
            return nn.ConvTranspose2d
    
    @classmethod
    def set_state(clss, state:str):
        assert state in [clss.TWO_D, clss.THREE_D], "Invalid state womp womp"
        clss.state = state
    
    @classmethod
    def avg_pool_op(clss):
        if clss.state == clss.THREE_D:
            return nn.AvgPool3d
        return nn.AvgPool2d


class XModule(nn.Module):
    """
    """

    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_sizes=[3], 
                 mode='concat', 
                 stride=1, 
                 **kwargs
                 ):
        super(XModule, self).__init__()
        self.branches = nn.ModuleList()

        if 'kernel_size' in kwargs:
            kernel_sizes = [kwargs['kernel_size']]

        assert mode in [CONCAT, ADD], "Valid values for mode are 'concat' and 'add'"
        self.mode = mode

        for k in kernel_sizes:
            assert (k-1) % 2 == 0, "kernel sizes must be odd numbers"
            if ModuleStateController.state == ModuleStateController.TWO_D:
                branch = self._get_2d_branch(k, in_channels, out_channels, stride)
            else:
                raise NotImplementedError("3D XModule not implemented")
            self.branches.append(branch)

        if mode == CONCAT:
            self.pw = nn.Sequential(
                nn.LeakyReLU(),
                ModuleStateController.conv_op()
                (in_channels=out_channels*len(kernel_sizes), out_channels=out_channels, kernel_size=1)
            )
    
    def _get_2d_branch(self, k:int, in_channels:int, out_channels:int, stride:int)->nn.Sequential:
        pad = (k-1)//2
        branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels,
                            kernel_size=(1, k), padding=(0, pad),
                            groups=out_channels, stride=(stride, stride)),
                nn.Conv2d(out_channels, out_channels,
                            kernel_size=(k, 1), padding=(pad, 0),
                            groups=out_channels),
            )
        return branch
    

    def forward(self, x):
        output = []
        for branch in self.branches:
            output.append(
                branch(x)
            )
        if self.mode == CONCAT:
            return self.pw(torch.concat(output, dim=1))
        else:
            return torch.sum(output)


class Linker(nn.Module):

    def __init__(self, mode:str, module:dict) -> None:
        """
        Can concatenate or add input for skipped connections before passing to a module.
        Used for JSON model architecture.
        """
        super().__init__()
        assert mode in [ADD, CONCAT]
        self.mode = mode
        self.module = ModelBuilder(module['Tag'], module['Children'])

    def forward(self, x:torch.Tensor, extra:torch.Tensor):
        extra = list(extra.values())
        assert len(extra) == 1, "Can only use Linker with a single extra input value."
        extra = extra[0]

        if self.mode == ADD:
            return self.module(x+extra)
        return self.module(torch.concat((x, extra), dim=1))
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_op:str, kernel_size=3, dilation=1, last_layer:bool=False, pad="default", transpose_kernel=2, transpose_stride=2, dropout_p:float=0.) -> None:
        super().__init__()

        conv_op = my_import(conv_op)
        transp_op = ModuleStateController.transp_op()
        norm_op = ModuleStateController.norm_op()

        pad = (kernel_size - 1) // 2 * dilation if pad == 'default' else int(pad)
        self.conv1 = nn.Sequential(
            conv_op(
                in_channels=in_channels*2,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad
            ),
            norm_op(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_op(
                in_channels=in_channels,
                out_channels=(out_channels if last_layer else in_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad
            )
        )
        if not last_layer:
            self.transpose = transp_op(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=transpose_stride,
                kernel_size=transpose_kernel
            )
            self.conv2.append(
                norm_op(num_features=out_channels)
            )
            self.conv2.append(
                nn.LeakyReLU(inplace=True)
            )
        
        self.last_layer = last_layer

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        if not self.last_layer:
            return self.transpose(x)
        return x

class PolyActivation(nn.Module):
    def __init__(self, order: int):
        super().__init__()
        self.order = order
    def forward(self, x):
        return torch.pow(x, self.order)

class PolyWrapper(nn.Module):
  def __init__(self, in_channels, out_channels, order, stride: int = 1):
    super().__init__()
    self.branches = nn.ModuleList([
        PolyBlock(in_channels, out_channels, o, stride) for o in range(1, order+1)
    ])
  def forward(self, x):
    out = None
    mean = torch.mean(x)
    x = torch.sub(x, mean)
    # x -= mean
    for mod in self.branches:
      if out is None:
        out = mod(x)
      else:
        out = torch.add(out, mod(x))
    return out

class PolyBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, order: int, stride: int = 1):
    super().__init__()
    self.order = order
    self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    self.ch_maxpool = nn.MaxPool3d((in_channels, 1, 1), stride=(in_channels, 1, 1))
  def forward(self, x):
    std = torch.std(x)
    # print(torch.max(x))
    x = torch.clip(x, -3*std, 3*std)
    # print(torch.max(x))
    x_pow = torch.pow(x, self.order)
    # print(torch.max(x_pow))
    norm = self.ch_maxpool(torch.abs(x_pow))
    x_normed = x_pow/(norm + 1e-7)
    # print(torch.max(x_normed))
    out = self.conv(x_normed)
    return out

class InstanceNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.norm = ModuleStateController.norm_op()(num_features=num_features)
    
    def forward(self, x):
        return self.norm(x)

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.transp_op = ModuleStateController.transp_op()(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride=stride)
    def forward(self, x):
        return self.transp_op(x)
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, padding='auto') -> None:
        super().__init__()
        padding = padding if isinstance(padding, int) else (kernel_size-1)//2
        self.conv = ModuleStateController.conv_op()(in_channels=in_channels, 
                                                    out_channels=out_channels, 
                                                    padding=padding, 
                                                    stride=stride, 
                                                    dilation=dilation,
                                                    kernel_size=kernel_size)
    def forward(self, x):
        return self.conv(x)

class AveragePool(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.pool = ModuleStateController.avg_pool_op()(kernel_size)
    def forward(self, x):
        return self.pool(x)

class MCDropout(nn.Module):
    def __init__(self, p:float=0.5) -> None:
        super().__init__()
        self.dropout = ModuleStateController.mcdropout_op()(p=p)
    def forward(self, x):
        return self.dropout(x)

class NonLinearity(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficiants = nn.ParameterList([nn.Parameter(torch.tensor([x], requires_grad=True)) for x in 
                                              [0., 1., 0., 0., 0.]])
        self.powers = nn.ParameterList([0., 1., 2., 3., 4.])
        import uuid
        self.id = str(uuid.uuid4())[0:5]
        self.iters = 0
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)
        result = torch.zeros_like(x)
        for co, pow in zip(self.coefficiants, self.powers):
            result = torch.add(result, torch.mul(co, torch.pow(x, pow)))
        if self.iters % 100 == 0:
            torch.save(self.powers, f"/home/andrew.heschl/Documents/599_Architecture_Project/nnunetv2/training/out/nonlin_p_{self.id}.pth")
            torch.save(self.coefficiants, f"/home/andrew.heschl/Documents/599_Architecture_Project/nnunetv2/training/out/nonlin_c_{self.id}.pth")
        self.iters += 1
        return self.relu(result)