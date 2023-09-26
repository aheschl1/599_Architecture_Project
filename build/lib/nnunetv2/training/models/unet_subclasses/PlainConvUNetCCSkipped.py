from typing import List, Tuple, Type, Union
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

class PlainConvUNetSkipOperationSkipped(PlainConvUNet):
    def __init__(self, input_channels: int, n_stages: int, features_per_stage: int | List[int] | Tuple[int, ...], conv_op: type[_ConvNd], kernel_sizes: int | List[int] | Tuple[int, ...], strides: int | List[int] | Tuple[int, ...], n_conv_per_stage: int | List[int] | Tuple[int, ...], num_classes: int, n_conv_per_stage_decoder: int | Tuple[int, ...] | List[int], conv_bias: bool = False, norm_op: type[nn.Module] | None = None, norm_op_kwargs: dict = None, dropout_op: type[_DropoutNd] | None = None, dropout_op_kwargs: dict = None, nonlin: type[nn.Module] | None = None, nonlin_kwargs: dict = None, deep_supervision: bool = False, nonlin_first: bool = False,
                 skipped_operation:nn.Module=None, skipped_operation_stages:list=[], skipped_operation_kwargs:map={}):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.decoder = PlainConvDecoderOperationSkipped(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first, 
                                                 skipped_operation, skipped_operation_stages, skipped_operation_kwargs)
        
class PlainConvDecoderOperationSkipped(UNetDecoder):
    def __init__(self, encoder: PlainConvEncoder | ResidualEncoder, num_classes: int, n_conv_per_stage: int | Tuple[int, ...] | List[int], deep_supervision, nonlin_first: bool = False, 
                 skipped_operation:nn.Module=None, skipped_operation_stages:list=[], skipped_operation_kwargs:map={}):
        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first)
        self.skipped_operation = skipped_operation(**skipped_operation_kwargs)
        self.skipped_operation_stages = skipped_operation_stages

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            
            skip = skips[-(s+2)]
            if s in self.skipped_operation_stages:
                print("my decoder")
                skip = self.skipped_operation(skip)

            x = torch.cat((x, skip), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
    
#====================================================================

class PlainConvUNetSkipOperationSkipped3d(PlainConvUNet):
    def __init__(self, input_channels: int, n_stages: int, features_per_stage: int | List[int] | Tuple[int, ...], conv_op: type[_ConvNd], kernel_sizes: int | List[int] | Tuple[int, ...], strides: int | List[int] | Tuple[int, ...], n_conv_per_stage: int | List[int] | Tuple[int, ...], num_classes: int, n_conv_per_stage_decoder: int | Tuple[int, ...] | List[int], conv_bias: bool = False, norm_op: type[nn.Module] | None = None, norm_op_kwargs: dict = None, dropout_op: type[_DropoutNd] | None = None, dropout_op_kwargs: dict = None, nonlin: type[nn.Module] | None = None, nonlin_kwargs: dict = None, deep_supervision: bool = False, nonlin_first: bool = False,
                 skipped_operation:nn.Module=None, skipped_operation_stages:list=[], skipped_operation_kwargs:map={}):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.decoder = PlainConvDecoderOperationSkipped3d(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first, 
                                                 skipped_operation, skipped_operation_stages, skipped_operation_kwargs)
        
class PlainConvDecoderOperationSkipped3d(UNetDecoder):
    def __init__(self, encoder: PlainConvEncoder | ResidualEncoder, num_classes: int, n_conv_per_stage: int | Tuple[int, ...] | List[int], deep_supervision, nonlin_first: bool = False, 
                 skipped_operation:nn.Module=None, skipped_operation_stages:list=[], skipped_operation_kwargs:map={}):
        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first)
        self.skipped_operation = skipped_operation(**skipped_operation_kwargs)
        self.skipped_operation_stages = skipped_operation_stages
    
    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            
            skip = skips[-(s+2)]
            if s in self.skipped_operation_stages:
                skip = self.skipped_operation(skips[-(s+2)])

            x = torch.cat((x, skip), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r