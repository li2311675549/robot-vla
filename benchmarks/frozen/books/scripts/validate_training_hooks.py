#!/usr/bin/env python3
"""CPU-only tiny-model plumbing test. Never loads or trains pi05 weights."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--lerobot-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve hook tests')
    os.environ['JAX_PLATFORMS']='cpu'
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(60,repeat=True)
    os.environ['HF_LEROBOT_HOME']=str(a.lerobot_cache.resolve())
    os.environ.setdefault('HF_DATASETS_CACHE',str(a.lerobot_cache.resolve().parent/'hf_datasets_cache'))
    os.environ['HF_HUB_OFFLINE']=os.environ['HF_DATASETS_OFFLINE']='1'
    sys.path.insert(0,str(a.openpi_root/'src'))
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import numpy as np
    import optax
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from robot_exam.vla.training_plan import TrainingPlan
    from robot_exam.vla.openpi_config import make_config
    from robot_exam.vla.openpi_validation import run_pinned_training
    from openpi.training import checkpoints,utils
    work=a.output.parent/(a.output.stem+'_work');work.mkdir(parents=True,exist_ok=False)
    plan=TrainingPlan.load(ROOT/'configs/pi05_dual_fruit_smoke_v1.json')
    cfg=make_config('local/robot_exam_dual_release_train',a.assets,Path('/tmp/no_weights'),work,
                    dry_check=True,wrist_camera=True,training_plan=plan)
    schedule=cfg.lr_schedule.create()
    rates=np.asarray([schedule(i) for i in range(plan.steps+1)])
    assert np.isfinite(rates).all() and rates[5]>rates[0] and rates[-1]<rates[5]

    class Tiny(nnx.Module):
        def __init__(self):self.bias=nnx.Param(jnp.zeros((7,)))
        def compute_loss(self,rng,observation,actions,train=False):
            del rng,observation,train
            return (actions-self.bias.value)**2
    graph,params=nnx.split(Tiny())
    state=utils.TrainState(step=jnp.array(0),params=params,model_def=graph,opt_state={},
                           tx=optax.sgd(.1),ema_decay=None)
    manager=None
    class Loader:
        def data_config(self):return cfg.data.create(cfg.assets_dirs,cfg.model)
    loader=Loader()
    module=SimpleNamespace(_checkpoints=checkpoints)
    module.init_train_state=lambda: (state,None)
    def fake_step(current):
        # JIT/callback check only: no parameter optimization, no real model.
        return dataclasses.replace(current,step=current.step+1),{'loss':jnp.array(1.),'grad_norm':jnp.array(0.)}
    module.train_step=fake_step
    def main_fn(config):
        nonlocal state,manager
        print('tiny model: validation dataset prepared; starting checkpoint/JIT test',flush=True)
        state,_=module.init_train_state()
        manager,_=checkpoints.initialize_checkpoint_dir(work/'tiny_checkpoint',keep_period=1,overwrite=False,resume=False)
        compiled=jax.jit(module.train_step)
        state,info=compiled(state);jax.block_until_ready(state)
        module._checkpoints.save_state(manager,state,loader,1)
    module.main=main_fn
    run_pinned_training(module,cfg,plan,a.lerobot_cache,work)
    restored=checkpoints.restore_state(manager,state,loader)
    assert int(restored.step)==1
    np.testing.assert_array_equal(restored.params['bias'].value,state.params['bias'].value)
    manager.close()
    rows=[json.loads(x) for x in (work/'validation.jsonl').read_text().splitlines()]
    assert rows[0]['sample_count']==26*plan.validation_frames_per_episode
    assert (work/'train_metrics.jsonl').is_file()
    assert json.loads((work/'trainable_parameters.json').read_text())['elements']==7
    assert (work/'tiny_checkpoint/1/assets/original_ur5/norm_stats.json').is_file()
    result={'status':'PASS_TINY_CPU_PLUMBING_ONLY','actual_validation_episodes':26,
            'actual_validation_samples':rows[0]['sample_count'],
            'checkpoint_save_restore':True,'jit_callback_logging':True,'finite_loss':True,
            'warmup_schedule_checked':True,'pi05_weights_loaded':False,'pi05_training_performed':False,
            'tiny_parameter_count':7,'parameter_optimization_performed':False,'work_directory':str(work)}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
