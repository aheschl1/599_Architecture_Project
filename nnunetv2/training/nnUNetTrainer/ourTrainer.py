import torch
from torch import autocast

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.models.model_generator import ModelGenerator
from nnunetv2.training.models.utils import ModuleStateController
from monai.losses import DiceCELoss, DiceLoss

class ourTrainer(nnUNetTrainer):

    loss_mode = "default"

    def set_model_path(self, model_path:str):
        self.model_path = model_path
        super().set_model_path(model_path)
    
    def set_log_path(self, log_path:str):
        self.log_path = log_path
        super().set_log_path(log_path)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def _build_loss(self):

        if ourTrainer.loss_mode == "monai":
            return DiceCELoss(
                            include_background=True,
                            batch=self.configuration_manager.batch_dice,
                            to_onehot_y=False,
                            sigmoid=True,
                            smooth_dr=1e-5,
                            smooth_nr=1e-5,
                        )

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        return loss

    def _get_deep_supervision_scales(self):
        return None

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
    
    def build_network_architecture(self, plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True):

        dims = len(configuration_manager.patch_size)
        assert dims in [2, 3], "How is the patch size not 2 or 3 dimensions?????"
        if dims == 2:
            ModuleStateController.set_state("2d")
            print("The module state has been set to 2d.")
        elif dims == 3:
            ModuleStateController.set_state("3d")
            print("The module state has been set to 3d. Make sure every module you use supports it.")

        generator = ModelGenerator(self.model_path)
        log_kwargs = generator.get_log_kwargs()
        if log_kwargs != None:
            super().print_to_log_file(log_kwargs)
            
        return generator.get_model()
    
    @staticmethod
    def build_network_architecture_static(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   model_path:str,
                                   enable_deep_supervision: bool = True):

        dims = len(configuration_manager.patch_size)
        assert dims in [2, 3], "How is the patch size not 2 or 3 dimensions?????"
        if dims == 2:
            ModuleStateController.set_state("2d")
            print("The module state has been set to 2d.")
        elif dims == 3:
            ModuleStateController.set_state("3d")
            print("The module state has been set to 3d. Make sure every module you use supports it.")

        generator = ModelGenerator(model_path)
        return generator.get_model()
    
