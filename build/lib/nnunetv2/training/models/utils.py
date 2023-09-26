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
    
    @classmethod
    def cc_link_op(clss):
        if clss.state == clss.THREE_D:
            return CrissCrossAttention3D
        return EfficientCC_Wrapper

    @classmethod
    def mcdropout_op(clss):
        if clss.state == clss.THREE_D:
            return MCDropout3D
        return MCDropout2D

class ChannelAttention_CAM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        #performing pooling operations

        conv_op = ModuleStateController.conv_op()

        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        #convolutions
        self.conv1by1 = conv_op(channels,channels//16,kernel_size = 1)
        self.conv1by1_2 = conv_op(channels//16,channels,kernel_size = 1)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        maxpooled = self.maxpooling(x)
        OutputMaxPooling = self.conv1by1(maxpooled)
        OutputMaxPooling = self.relu(OutputMaxPooling)
        OutputMaxPooling = self.conv1by1_2(OutputMaxPooling)
        
        avgpooled = self.avgpooling(x)
        OutputAvgPooling = self.conv1by1(avgpooled)
        OutputAvgPooling = self.relu(OutputAvgPooling)
        OutputAvgPooling = self.conv1by1_2(OutputAvgPooling)
        
        #element wise summation
        outputFeatureMap = OutputMaxPooling + OutputAvgPooling
        ftrMap = self.sigmoid(outputFeatureMap)
        ftr = ftrMap * x
        return ftr

class SpatialAttention_CAM(nn.Module):
    def __init__(self):
        super().__init__()

        conv_op = ModuleStateController.conv_op()
        #performing channelwise pooling
        self.spatialmaxpool = Reduce('b c h w -> b 1 h w', 'max')
        self.spatialavgpool = Reduce('b c h w -> b 1 h w', 'mean')
        #padding to keep the tensor shape same as input
        self.conv = conv_op(1,1,kernel_size = 7,padding = 3)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        maxpooled = self.spatialmaxpool(x)
        #print(maxpooled.shape)
        avgpooled = self.spatialavgpool(x)
        #print(avgpooled.shape)
        #adding the tensors
        summed = maxpooled + avgpooled
        #print(summed.shape)
        convolved = self.conv(summed)
        #print(convolved.shape)
        ftrMap = self.sigmoid(convolved)
        #print(ftrMap.shape)
        ftr = ftrMap * x
        return ftr

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        conv_op = ModuleStateController.conv_op()
        norm_op = ModuleStateController.norm_op()

        self.conv = conv_op(channels,channels,kernel_size = 3, padding=1)
        self.bn = norm_op(channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        #print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        #print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention_CAM(channels)
        self.spatial=SpatialAttention_CAM()
        self.conv = ConvBlock(channels)
    def forward(self,x):
        x = self.channel(x)
        x = self.spatial(x)
        x = self.conv(x)
        return x

class LearnableChannelAttention_CAM(nn.Module):
    def __init__(self,channels, mode=0):
        super().__init__()
        #performing pooling operations
        self.f_0 = nn.Sequential()
        self.f_1 = nn.Sequential()
        self.mode = mode

        for i in range(2):
            pool = nn.AdaptiveAvgPool2d(8) if i == 0 else nn.AdaptiveMaxPool2d(8)
            conv = DepthWiseSeparableConv(channels, channels, kernel_sizes=[8], pad=0, use_norm=False)
            
            for module in [pool, nn.ReLU(inplace=True), conv, nn.ReLU(inplace=True)]:
                if i == 0:
                    self.f_0.append(module)
                else: 
                    self.f_1.append(module)
        
    def forward(self,x):
        output_0 = self.f_0(x)
        output_1 = self.f_1(x)

        if self.mode == 0:
            x = torch.add(output_0, output_1)
            return torch.mul(nn.Sigmoid()(x), x)
        else:
            y = torch.add(torch.mul(nn.Sigmoid()(output_0), x), torch.mul(nn.Sigmoid()(output_1), x))
            return torch.add(x, y)
class LearnableCAM(nn.Module):
    def __init__(self, channels, mode = 0):
        super().__init__()
        self.channel = LearnableChannelAttention_CAM(channels, mode=mode)
    def forward(self,x):
        ax = self.channel(x)
        return torch.add(x, ax)
    
#CBAM start=================================
class LearnableChannelAttention(nn.Module):
    def __init__(self, channels, r, dimension):
        super().__init__()
        #performing pooling operations
        self.pool = nn.MaxPool2d(2)
        dimension//=2 #Because pool
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(dimension, dimension), groups=channels)
        #input the results of pooling to the 1 hidden layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels//r),
            nn.ReLU(),
            nn.Linear(channels//r, channels))
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        reduced = self.pool(x)
        convolved = self.conv(reduced)
        #squeeze to reduce dimension (n,c,1,1) to (n,c)
        convolved = torch.squeeze(convolved)
        outputFeatureMap = self.mlp(convolved)
        #element wise summation
        ftrMap = self.sigmoid(outputFeatureMap)
        #print(ftrMap.shape)
        #converting tension (n,c) to (n,c,w,h)
        ftrMap = ftrMap.unsqueeze(-1)
        ftrMap = ftrMap.unsqueeze(-1)
        #print(ftrMap.shape)
        ftr = ftrMap * x
        #print(ftr.shape)
        return ftr

class ChannelAttention(nn.Module):
    def __init__(self,channels,r):
        super().__init__()
        #performing pooling operations
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        #input the results of pooling to the 1 hidden layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels//r),
            nn.ReLU(),
            nn.Linear(channels//r, channels))
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        maxpooled = self.maxpooling(x)
        #squeeze to reduce dimension (n,c,1,1) to (n,c)
        maxpooled = torch.squeeze(maxpooled)
        #print(maxpooled.shape)
        avgpooled = self.avgpooling(x)
        #squeeze to reduce dimension (n,c,1,1) to (n,c)
        avgpooled = torch.squeeze(avgpooled)
        #print(avgpooled.shape)
        mlpOutputMaxPooling = self.mlp(maxpooled)
        mlpOutputAvgPooling = self.mlp(avgpooled)
        #element wise summation
        outputFeatureMap = mlpOutputMaxPooling + mlpOutputAvgPooling
        ftrMap = self.sigmoid(outputFeatureMap)
        #print(ftrMap.shape)
        #converting tension (n,c) to (n,c,w,h)
        ftrMap = ftrMap.unsqueeze(-1)
        ftrMap = ftrMap.unsqueeze(-1)
        #print(ftrMap.shape)
        ftr = ftrMap * x
        #print(ftr.shape)
        return ftr

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        #performing channelwise pooling
        self.spatialmaxpool = Reduce('b c h w -> b 1 h w', 'max')
        self.spatialavgpool = Reduce('b c h w -> b 1 h w', 'mean')
        #padding to keep the tensor shape same as input
        self.conv1d = nn.Conv2d(2,1,kernel_size = 7,padding = 3)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        maxpooled = self.spatialmaxpool(x)
        #print(maxpooled.shape)
        avgpooled = self.spatialavgpool(x)
        #print(avgpooled.shape)
        #concatenating the tensors
        concat = torch.cat([maxpooled, avgpooled], dim=1)
        #print(concat.shape)
        convolved = self.conv1d(concat)
        #print(convolved.shape)
        ftrMap = self.sigmoid(convolved)
        #print(ftrMap.shape)
        ftr = ftrMap * x
        return ftr

class CBAM(nn.Module):
    def __init__(self, channels, r, stride=1, mode = "regular", dimension=-1):
        super().__init__()
        assert mode in["regular", "learnable"], "The two modes are 'learnable' and 'regular'."
        assert mode == 'regular' or dimension != -1, "If the mode is 'learnable' specify the dimension parameter."
        self.channel = ChannelAttention(channels,r) if mode == "regular" else LearnableChannelAttention(channels, r, dimension)
        self.spatial=SpatialAttention()
    def forward(self,x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class UpsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 upscale_factor=2, mode='bilinear'):
        super(UpsamplingConv, self).__init__()
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode=mode, align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding)
        )

    def forward(self, x):
        return self.module(x)

#CBAM end===========================================================

class ConvPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 upscale_factor=2):
        super(ConvPixelShuffle, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * upscale_factor ** 2,
                      kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
            return self.module(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, attention_channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.out = nn.Conv2d(attention_channels, in_channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x):
        batch_size, _, height, width = x.size()
        query = self.query(x).view(batch_size, self.num_heads, -1, height * width).permute(0, 2, 1, 3)
        key = self.key(x).view(batch_size, self.num_heads, -1, height * width)
        value = self.value(x).view(batch_size, self.num_heads, -1, height * width).permute(0, 2, 1, 3)

        attention_weights = torch.matmul(query, key) / math.sqrt(value.size(-2))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        attended_values = torch.matmul(attention_weights, value).permute(0, 2, 1, 3)
        attended_values = attended_values.contiguous().view(batch_size, -1, height, width)

        return self.out(attended_values) + x


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilations_dsc=[1], 
                 kernel_sizes_dsc=[3], mode='concat',
                 stride=1, padding='default', use_norm = False, **kwargs
                 ):
        if 'kernel_size' in kwargs:
            kernel_sizes_dsc = [kwargs['kernel_size']]
             
        assert len(dilations_dsc) == len(kernel_sizes_dsc)
        assert mode in ['concat', 'add']

        self.mode = mode
        #GET OPERATIONS
        norm = ModuleStateController.norm_op()
        conv_op = ModuleStateController.conv_op()
        
        super(DepthWiseSeparableConv, self).__init__()
        self.branches = nn.ModuleList()
        for dilation, kernel_size in zip(dilations_dsc, kernel_sizes_dsc):
            pad = (kernel_size - 1) // 2 * dilation if padding == 'default' else int(padding)
            branch = nn.Sequential(
                conv_op(in_channels, in_channels, kernel_size=kernel_size, padding=pad,
                          dilation=dilation, groups=in_channels, stride=stride),
                conv_op(in_channels, out_channels, kernel_size=1),
            )
            
            if use_norm:
                branch.insert(1, norm(num_features=in_channels))
                
            self.branches.append(branch)

    def forward(self, x):
        results = []
        for branch in self.branches:
            results.append(branch(x))
        if self.mode == 'concat':
            return torch.concat(tuple(results), dim=1)
        return torch.sum(torch.stack(results), dim=0)


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()

        conv_op = ModuleStateController.conv_op()

        self.depthwise_conv = conv_op(in_channels, in_channels,
                                        kernel_size=1, groups=in_channels)
        self.pointwise_conv = conv_op(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.depthwise_conv(x)
        attention_map = self.pointwise_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        out = x * attention_map
        return out

class AttentionX(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, kernel_sizes, pool=False, mode=CONCAT, stride=1):

        super().__init__()
        self.branches = nn.ModuleList()
        assert len(dilations) == len(kernel_sizes) == 2, "Only two branches can be applied"
        assert mode in [CONCAT, ADD]
        assert kernel_sizes[0] % 2 == 1 and kernel_sizes[1]%2==1
        self.mode = mode

        pad = (kernel_sizes[0]-1)//2 * dilations[0]
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, kernel_sizes[0]), padding=(0, pad), stride=stride, dilation=dilations[0]),
            nn.Conv2d(out_channels, out_channels, (kernel_sizes[0], 1), padding=(pad, 0), dilation=dilations[0]),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        pad = (kernel_sizes[1]-1)//2 * dilations[1]
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, kernel_sizes[1]), padding=(0, pad), stride=stride, dilation=dilations[1]),
            nn.Conv2d(out_channels, out_channels, (kernel_sizes[1], 1), padding=(pad, 0), dilation=dilations[1]),
            nn.LeakyReLU()
        )
        #This mask will contain information on what should be kept from the low res convolution.
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding = 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(out_channels*2 if mode == CONCAT else out_channels, out_channels, kernel_size=1)
        )
        if pool:
            self.pointwise.append(nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        top_conv = self.branch_0(x)
        bottom_conv = self.branch_1(x)
        mask = self.attention(top_conv)
        top_conv = torch.mul(top_conv, mask)
        bottom_conv = torch.mul(1-mask, bottom_conv)

        if self.mode == CONCAT:
            return self.pointwise(torch.cat((top_conv, bottom_conv), dim=1))
        
        return self.pointwise(torch.add(top_conv, bottom_conv))
        

class XModule(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, dilations=[1], kernel_sizes=[3], mode='concat', stride=1, apply_norm:bool=False, gated=False, **kwargs):
        super(XModule, self).__init__()
        self.branches = nn.ModuleList()
        self.apply_norm = apply_norm

        assert not(ModuleStateController.state == '3d' and gated), "Cannot use gated convolution in 3d XModule."
        self.gated = gated

        if 'kernel_size' in kwargs:
            kernel_sizes = [kwargs['kernel_size']]

        assert len(dilations) == len(kernel_sizes)
        assert mode in [CONCAT, ADD], "Valid values for mode are 'concat' and 'add'"
        assert out_channels % len(dilations) == 0, f"Got out channels: {out_channels}"
        self.mode = mode

        for d, k in zip(dilations, kernel_sizes):
            assert (k-1) % 2 == 0, "kernel sizes must be odd numbers"
            if ModuleStateController.state == ModuleStateController.TWO_D:
                branch = self._get_2d_branch(d, k, in_channels, out_channels, stride)
            else:
                if "heavy" in kwargs and kwargs['heavy']:
                    branch = self._get_3d_branch_heavy(d, k, in_channels, out_channels, stride)
                else:
                    branch = self._get_3d_branch(d, k, in_channels, out_channels, stride)

            self.branches.append(branch)

        if mode == CONCAT:
            self.pw = nn.Sequential(
                nn.LeakyReLU(),
                ModuleStateController.conv_op()(in_channels=out_channels*len(dilations), out_channels=out_channels, kernel_size=1)
            )
    
    def _get_2d_branch(self, d:int, k:int, in_channels:int, out_channels:int, stride:int)->nn.Sequential:
        pad = (k-1)//2 * d
        first_op = nn.Conv2d if not self.gated else SpatialGatedConv2d
        branch = nn.Sequential(
                first_op(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels,
                            kernel_size=(1, k), dilation=d, padding=(0, pad),
                            groups=out_channels, stride=(stride, stride)),
                nn.Conv2d(out_channels, out_channels,
                            kernel_size=(k, 1), dilation=d, padding=(pad, 0),
                            groups=out_channels),
            )
        
        if self.apply_norm:
            branch.insert(1, nn.InstanceNorm2d(num_features=out_channels))
        return branch
    
    def _get_3d_branch(self, d:int, k:int, in_channels:int, out_channels:int, stride:int)->nn.Sequential:
        pad = (k-1)//2 * d
        branch = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.Conv3d(out_channels, out_channels,
                            kernel_size=(k, 1, 1), dilation=d, padding=(pad, 0, 0),
                            groups=out_channels, stride=stride),
                nn.Conv3d(out_channels, out_channels,
                            kernel_size=(1, k, 1), dilation=d, padding=(0, pad, 0),
                            groups=out_channels),
                nn.Conv3d(out_channels, out_channels,
                            kernel_size=(1, 1, k), dilation=d, padding=(0, 0, pad),
                            groups=out_channels),
            )
        
        if self.apply_norm:
            branch.insert(1, nn.InstanceNorm3d(num_features=out_channels))
        return branch

    def _get_3d_branch_heavy(self, d:int, k:int, in_channels:int, out_channels:int, stride:int)->nn.Sequential:
        pad = (k-1)//2 * d
        branch = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.Conv3d(out_channels, out_channels,
                            kernel_size=(k, k, 1), dilation=d, padding=(pad, pad, 0),
                            groups=out_channels, stride=stride),
                nn.Conv3d(out_channels, out_channels,
                            kernel_size=(1, 1, k), dilation=d, padding=(0, 0, pad),
                            groups=out_channels),
            )
        
        if self.apply_norm:
            branch.insert(1, nn.InstanceNorm3d(num_features=out_channels))
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

class CBAMResidual(nn.Module):
    def __init__(self, module:dict, channels:int, r:int, mode='concat'):
        """
          ---------------->
         |                 |
         |                 |---(concat or add along channels)-->
        [in]----(module)-->
        """
        super().__init__()
        self.module = ModelBuilder(module['Tag'], module['Children'])
        self.mode = mode
        self.cbam = CBAM(channels=channels, r=r)

    def forward(self, x):
        out = self.module(x)
        if self.mode == CONCAT:
            assert x.shape[2:] == out.shape[2:], \
                f'module must create the shape [B, -1, height_x, width_x] when concating. Expected shape[2:] {x.shape[2:]}, got {out.shape[2:]}'
            return torch.concat((self.cbam(x), out), dim=1)
        
        assert out.shape == x.shape, f'module must create the shape [B, C, height_x, width_x] when adding. Expected {x.shape}, got {out.shape}'
        return torch.add(out, self.cbam(x))

class Residual(nn.Module):
    def __init__(self, module:dict, mode='concat'):
        """
          ---------------->
         |                 |
         |                 |---(concat or add along channels)-->
        [in]----(module)-->
        """
        super().__init__()
        self.module = ModelBuilder(module['Tag'], module['Children'])
        self.mode = mode

    def forward(self, x):
        out = self.module(x)
        if self.mode == CONCAT:
            assert x.shape[2:] == out.shape[2:], \
                f'module must create the shape [B, -1, height_x, width_x] when concating. Expected shape[2:] {x.shape[2:]}, got {out.shape[2:]}'
            return torch.concat((x, out), dim=1)
        
        assert out.shape == x.shape, f'module must create the shape [B, C, height_x, width_x] when adding. Expected {x.shape}, got {out.shape}'
        return torch.add(out, x)


class ChannelGroupAttention(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        assert num_channels % num_groups == 0
        self.scale_factor = num_channels // num_groups

        # G is a learnable parameter
        self.G = nn.Parameter(torch.rand(num_groups, num_groups))
        """ Element of real A x B, A = B = num groups. Should esentially represent that channel group R is similar to channel group C.
             __________
            |
            |
            |
            |
        """
    def forward(self, x):
        # Expand G to build C, which is the expanded version of G with each
        # element of G being repeated num_channels % num_groups times
        with torch.no_grad():
            C = self.G.repeat_interleave(self.scale_factor, dim=0).repeat_interleave(self.scale_factor, dim=1)

        """ Element of real A x B, A = B = num channels. C is not learnable
             __________
            |
            |
            |
            |
        """
        #print(C.shape, C.requires_grad, self.G.shape) [[channels, channels], False, [Groups, Groups]]

        num_batches, num_channels, height, width = x.shape
        # Flatten the spatial dimensions of the input tensor
        x = x.view(num_batches, num_channels, -1) #(B, C, H x W)
        # Now, transpose x to have dimensions (num_channels, -1, num_channels)
        x = x.transpose(1, 2)          #TODO:Check that this does what it is supposed to, because the ouptut is actually [batch, h*w, channels], not (channels, -1, channels) as commented.
        # Perform the matrix multiplication
        x = torch.matmul(x, C) 
        """
        h x w = n
        [n x c][c x c] -> [n x c] ... x retains shape
        """
        #print(x.shape)
        # Finally, transpose back the output tensor to the original form
        x = x.transpose(1, 2)
        # Reshape x back to the original shape
        x = x.view(num_batches, num_channels, height, width) #(b, c, h, w)
        return x
    
def INF(B,H,W, device):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
     
class CC_module(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x




class EfficientCC_module(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, recurrence):
        #print(x.shape)
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        if recurrence % 2 == 1:
            proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
            proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
            proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
            energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
            score = self.softmax(energy_W)
            att_W = score[:,:,:,:].contiguous().view(m_batchsize*height,width,width)
            out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
            return self.gamma*(out_W) + x

        else:
            proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
            proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
            proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
            energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
            score = self.softmax(energy_H)
            att_H = score[:,:,:,:].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
            out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
            return self.gamma*(out_H) + x

class EfficientCC_Wrapper(nn.Module):
    def __init__(self, channels, recurrence=2) -> None:
        super().__init__()
        assert recurrence%2 == 0
        self.recurrence = recurrence
        self.criss_cross = EfficientCC_module(channels = channels)
    
    def forward(self, x):
        for i in range(self.recurrence):
            x = self.criss_cross(x, i)
        return x

class SpatialGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=None, stride=1, kernel_size=1, dilation=1, padding=1, **kwargs):
        super(SpatialGatedConv2d, self).__init__()

        self.conv_gate = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size-1)//2)
        self.conv_values = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size-1)//2)
        self.nonlinearity = nonlinearity

        # Create coordinate map as a constant parameter
        self.coord = None

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # Generate coordinate maps if not created yet.
        if self.coord is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
            grid_x = grid_x.to(x.device) / (width - 1)
            grid_y = grid_y.to(x.device) / (height - 1)
            coord_single = torch.concat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0).unsqueeze(0)
            coord = coord_single
            for _ in range(batch_size-1):
                coord = torch.concat((coord, coord_single), dim=0)
            self.coord = coord
        # Concatenate coordinates with the input
        x_with_coords = torch.cat([x, self.coord], dim=1)

        gate = torch.sigmoid(self.conv_gate(x_with_coords))
        values = self.conv_values(x_with_coords)

        output = gate * values

        if self.nonlinearity is not None:
            output = self.nonlinearity(output)

        return output

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
    
class CC_Linker(nn.Module):

    def __init__(self, mode:str, module:dict, channels:int) -> None:
        """
        Can concatenate or add input for skipped connections before passing to a module.
        Used for JSON model architecture.
        """
        super().__init__()
        assert mode in [ADD, CONCAT]
        self.mode = mode
        self.module = ModelBuilder(module['Tag'], module['Children'])
        self.cc = ModuleStateController.cc_link_op(channels)
    def forward(self, x:torch.Tensor, extra:torch.Tensor):
        extra = list(extra.values())
        assert len(extra) == 1, "Can only use Linker with a single extra input value."
        extra = self.cc(extra[0])

        if self.mode == ADD:
            return self.module(x+extra)
        return self.module(torch.concat((x, extra), dim=1))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_op:str, kernel_size=3, dilation=1, last_layer:bool=False, pad="default", transpose_kernel=2, transpose_stride=2, dropout_p:float=0.) -> None:
        super().__init__()

        conv_op = my_import(conv_op)
        transp_op = ModuleStateController.transp_op()
        norm_op = ModuleStateController.norm_op()
        dropout_op = ModuleStateController.mcdropout_op()

        pad = (kernel_size - 1) // 2 * dilation if pad == 'default' else int(pad)
        self.conv1 = nn.Sequential(
            conv_op(
                in_channels=in_channels*2,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad
            ),
            dropout_op(dropout_p),
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
                dropout_op(dropout_p)
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

def INF3DH(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W*D,1,1)#.cuda()
def INF3DW(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(W),0).unsqueeze(0).repeat(B*H*D,1,1)#.cuda()
def INF3DD(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(D),0).unsqueeze(0).repeat(B*H*W,1,1)#.cuda()

class CrissCrossAttention3D(nn.Module):
    """ Criss-Cross Attention Module 3D version, inspired by the 2d version, but 3D CC Module should mask out the overlapped elements twice!"""
    def __init__(self, channels, verbose = False):
        super(CrissCrossAttention3D,self).__init__()
        in_dim = channels
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=4)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.verbose = verbose
        self.INFH = INF3DH
        self.INFD = INF3DD


    #def forward(self, proj_query,proj_key,proj_value):
    def forward(self, x):
        m_batchsize, _, height, width, depth= x.size()#proj_query.size()
        proj_query = self.query_conv(x)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize*width*depth,-1,height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize*height*depth,-1,width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize*height*width,-1,depth).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)

        # bchw > bwch, b*w*d-c-h
        proj_key_H = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
        # bchw > bhcw, b*h*d-c-w
        proj_key_W = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
        proj_key_D = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#b*h*w-c-d

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)#bchwd->bwdch
        proj_value_W = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)#bchwd->bhdcw
        proj_value_D = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#bchwd->bhwcd

        # batch matrix-matrix
        inf_holder = self.INFH(m_batchsize, height, width, depth).to(x.device) # > bw-h-h 
        energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize,width,depth,height,height).permute(0,1,3,2,4) # bwhdh

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(proj_query_W, proj_key_W)#+self.INFW(m_batchsize, height, width, depth)
        energy_W = energy_W.view(m_batchsize, height, depth, width, width).permute(0, 3, 1, 2, 4) # bwhdw
        
        #  b*h*w-d-c, b*h*w-c-d > b*h*w-d-d
        energy_D = (torch.bmm(proj_query_D, proj_key_D)+self.INFD(m_batchsize, height, width, depth).to(x.device)).view(m_batchsize, height, width, depth, depth).permute(0, 2, 1, 3, 4)# bwhdd


        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4)) # bwhd*(h+w+d)
        # bhw(H+W) > bhwH, bwhH; 
        att_H = concate[:,:,:,:,0:height].permute(0,1,3,2,4).contiguous().view(m_batchsize*width*depth,height,height)
        att_W = concate[:,:,:,:,height:height+width].permute(0,2,3,1,4).contiguous().view(m_batchsize*height*depth,width,width)
        att_D = concate[:,:,:,:,height+width:].permute(0,2,1,3,4).contiguous().view(m_batchsize*height*width, depth, depth)


        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,depth,-1,height).permute(0,3,4,1,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,depth,-1, width).permute(0,3,1,4,2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize,height, width, -1, depth).permute(0,3,1,2,4)

        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W + out_D) + x

class CC_Wrapper3d(nn.Module):
    def __init__(self, channels, recurrence=2) -> None:
        super().__init__()
        assert recurrence%2 == 0
        self.recurrence = recurrence
        self.criss_cross = CrissCrossAttention3D(channels = channels)
    
    def forward(self, x):
        for _ in range(self.recurrence):
            x = self.criss_cross(x)
        return x

class MCDropout(nn.Module):
    def __init__(self, p:float=0.5) -> None:
        super().__init__()
        self.dropout = ModuleStateController.mcdropout_op()(p=p)
    def forward(self, x):
        return self.dropout(x)