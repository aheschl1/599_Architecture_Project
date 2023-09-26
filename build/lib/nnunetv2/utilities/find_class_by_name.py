import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr

def my_import(class_name:str, dropout_package:str = 'torch.nn'):
    if class_name == 'UpsamplingConv':
        from nnunetv2.training.models.utils import UpsamplingConv
        return UpsamplingConv
    elif class_name == 'ConvPixelShuffle':
        from nnunetv2.training.models.utils import ConvPixelShuffle
        return ConvPixelShuffle
    elif class_name == 'SelfAttention':
        from nnunetv2.training.models.utils import SelfAttention
        return SelfAttention
    elif class_name == 'SpatialAttentionModule':
        from nnunetv2.training.models.utils import SpatialAttentionModule
        return SpatialAttentionModule
    elif class_name == 'DepthWiseSeparableConv':
        from nnunetv2.training.models.utils import DepthWiseSeparableConv
        return DepthWiseSeparableConv
    elif class_name == "MyConvTranspose2d":
        from nnunetv2.training.models.utils import MyConvTranspose2d
        return MyConvTranspose2d
    elif class_name == "MyConv2d":
        from nnunetv2.training.models.utils import MyConv2d
        return MyConv2d
    elif class_name == "Residual":
        from nnunetv2.training.models.utils import Residual
        return Residual
    elif class_name == "Linker":
        from nnunetv2.training.models.utils import Linker
        return Linker
    elif class_name == "XModule":
        from nnunetv2.training.models.utils import XModule
        return XModule
    elif class_name == "XModule_Norm":
        from nnunetv2.training.models.utils import XModule_Norm
        return XModule_Norm
    elif class_name == "AttentionX":
        from nnunetv2.training.models.utils import AttentionX
        return AttentionX
    elif class_name == "XModuleInverse":
        from nnunetv2.training.models.utils import XModuleInverse
        return XModuleInverse
    elif class_name == "CBAM":
        from nnunetv2.training.models.utils import CBAM
        return CBAM
    elif class_name == "CBAMResidual":
        from nnunetv2.training.models.utils import CBAMResidual
        return CBAMResidual
    elif class_name == "XModuleNoPW":
        from nnunetv2.training.models.utils import XModuleNoPW
        return XModuleNoPW
    elif class_name == "CAM":
        from nnunetv2.training.models.utils import CAM
        return CAM
    elif class_name == "XModuleReluBetween":
        from nnunetv2.training.models.utils import XModuleReluBetween
        return XModuleReluBetween
    elif class_name == "XModuleNoPWReluBetween":
        from nnunetv2.training.models.utils import XModuleNoPWReluBetween
        return XModuleNoPWReluBetween
    elif class_name == "LearnableCAM":
        from nnunetv2.training.models.utils import LearnableCAM
        return LearnableCAM
    elif class_name == "EfficientCC_Wrapper":
        from nnunetv2.training.models.utils import EfficientCC_Wrapper
        return EfficientCC_Wrapper
    elif class_name == "CC_module":
        from nnunetv2.training.models.utils import CC_module
        return CC_module
    elif class_name == "DecoderBlock":
        from nnunetv2.training.models.utils import DecoderBlock
        return DecoderBlock
    elif class_name == "InstanceNorm":
        from nnunetv2.training.models.utils import InstanceNorm
        return InstanceNorm
    elif class_name == "ConvTranspose":
        from nnunetv2.training.models.utils import ConvTranspose
        return ConvTranspose
    elif class_name == "Conv":
        from nnunetv2.training.models.utils import Conv
        return Conv
    elif class_name == "SpatialGatedConv2d":
        from nnunetv2.training.models.utils import SpatialGatedConv2d
        return SpatialGatedConv2d
    elif class_name == "AveragePool":
        from nnunetv2.training.models.utils import AveragePool
        return AveragePool
    elif class_name == "CC_Wrapper3d":
        from nnunetv2.training.models.utils import CC_Wrapper3d
        return CC_Wrapper3d
    elif class_name == "CC_Linker":
        from nnunetv2.training.models.utils import CC_Linker
        return CC_Linker
    elif class_name == "MCDropout":
        from nnunetv2.training.models.utils import MCDropout
        return MCDropout 
        
    else:
        module = importlib.import_module(dropout_package)
        class_ = getattr(module, class_name)
        return class_