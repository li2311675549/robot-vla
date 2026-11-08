"""Local OpenPI factory. Import only in the existing Python 3.11 OpenPI env."""
import dataclasses
from pathlib import Path

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config,weight_loaders

from robot_exam.vla.interface import OriginalUR5Inputs,OriginalUR5Outputs,ACTION_SEMANTICS


@dataclasses.dataclass(frozen=True)
class OriginalUR5DataConfig(config.DataConfigFactory):
    # Dry checks must not instantiate tokenizers or fetch assets.
    dry_check: bool = False
    wrist_camera: bool = False

    def create(self,assets_dirs,model_config):
        base=self.create_base_config(assets_dirs,model_config)
        mapping={'state':'observation.state','front_rgb':'observation.images.front',
                 'actions':'action','prompt':'prompt'}
        if self.wrist_camera:mapping['wrist_rgb']='observation.images.wrist'
        return dataclasses.replace(base,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(mapping)]),
            data_transforms=transforms.Group(inputs=[OriginalUR5Inputs(require_wrist=self.wrist_camera)],outputs=[OriginalUR5Outputs()]),
            model_transforms=transforms.Group() if self.dry_check else config.ModelTransformFactory()(model_config),
            action_sequence_keys=('action',),prompt_from_task=True)


def make_config(repo_id,assets_dir,weights_dir,checkpoint_dir,dry_check=False,wrist_camera=False):
    # Preserve the base model's padded 32D parameter shapes. Only the first
    # seven channels carry robot actions; horizon 10 is two seconds at 5 Hz.
    model=pi0_config.Pi0Config(pi05=True,action_dim=32,action_horizon=10,
        paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    return config.TrainConfig(name='pi05_original_ur5_lora_v1',exp_name='short_skill_pilot',
        model=model,data=OriginalUR5DataConfig(repo_id=repo_id,
            assets=config.AssetsConfig(assets_dir=str(Path(assets_dir).resolve()),asset_id='original_ur5'),
            base_config=config.DataConfig(prompt_from_task=True),dry_check=dry_check,wrist_camera=wrist_camera),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(weights_dir).resolve())),
        freeze_filter=model.get_freeze_filter(),ema_decay=None,batch_size=2,num_workers=0,
        num_train_steps=1000,save_interval=250,log_interval=10,
        checkpoint_base_dir=str(Path(checkpoint_dir).resolve()),wandb_enabled=False,
        policy_metadata={'robot':'original_ur5_2f85','action_semantics':ACTION_SEMANTICS,
                         'policy_hz':5,'actual_cameras':['front','wrist'] if wrist_camera else ['front'],
                         'wrist_cameras':wrist_camera})
