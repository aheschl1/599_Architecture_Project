from nnunetv2.training.models.utils import *
from nnunetv2.training.models.model_generator import ModelGenerator
import glob
mode = str(input("Mode: "))
assert mode in ['2d', '3d']

ModuleStateController.set_state(mode)

if mode == "3d":
    x = torch.rand((1, 1, 128, 128, 128))
else:
    x = torch.rand((56, 1, 256, 256))

for path in glob.glob('/home/andrewheschl/Documents/Seg3D/nnunetv2/training/models/mc_dropout_models/xmodules/**/*.json', recursive=True):
    module = ModelGenerator(path)
    #summary(module.get_model().cuda(), (1, 128, 128, 128))
    try:
        print(f"Success {path}: {module.get_model()(x).shape}")
    except:
        print(f"Fail: {path}")
        module.get_model()(x)
