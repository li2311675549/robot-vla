#!/usr/bin/env python3
"""Prepared inference server; requires a real user-provided trained checkpoint."""
import argparse
import os
import json
import time
from pathlib import Path
import sys
from _bootstrap import ROOT


def evaluation_noise_seed(headers, enabled):
    raw = headers.get('X-Robot-Exam-Evaluation-Seed')
    if raw is None:
        return 0
    if not enabled:
        raise ValueError('Per-episode noise control is not enabled')
    value = int(raw)
    if not 0 <= value < 2**31:
        raise ValueError('Evaluation noise seed is out of range')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--serve',action='store_true')
    p.add_argument('--repo-id',default='local/robot_exam_dual_release_train');p.add_argument('--wrist-camera',action='store_true')
    p.add_argument('--tokenizer',type=Path);p.add_argument('--lerobot-cache',type=Path)
    p.add_argument('--project-gripper',action='store_true',help='Versioned bounded gripper projection; strict remains default')
    p.add_argument('--websocket-compression',choices=('deflate',))
    p.add_argument('--evaluation-noise-seeds',action='store_true',help='Explicit inference-only RNG control via transport header, never model observations')
    a=p.parse_args()
    os.environ['HF_HUB_OFFLINE']=os.environ['HF_DATASETS_OFFLINE']='1'
    if a.lerobot_cache:
        os.environ['HF_LEROBOT_HOME']=str(a.lerobot_cache.resolve())
        os.environ.setdefault('HF_DATASETS_CACHE',str(a.lerobot_cache.resolve().parent/'hf_datasets_cache'))
    sys.path.insert(0,str(a.openpi_root/'src'))
    from robot_exam.vla.openpi_config import make_config
    cfg=make_config(a.repo_id,a.assets,a.checkpoint,ROOT/'checkpoints',dry_check=not a.serve,wrist_camera=a.wrist_camera)
    if a.project_gripper:
        import dataclasses
        from openpi import transforms
        from robot_exam.vla.openpi_config import OriginalUR5DataConfig
        from robot_exam.vla.policy_safety import ProjectGripperOutputs
        @dataclasses.dataclass(frozen=True)
        class ProjectedServingData(OriginalUR5DataConfig):
            def create(self,assets_dirs,model_config):
                data=super().create(assets_dirs,model_config)
                return dataclasses.replace(data,data_transforms=transforms.Group(
                    inputs=data.data_transforms.inputs,outputs=[ProjectGripperOutputs()]))
        factory=ProjectedServingData(**{f.name:getattr(cfg.data,f.name) for f in dataclasses.fields(cfg.data)})
        cfg=dataclasses.replace(cfg,data=factory,policy_metadata={**cfg.policy_metadata,
            'gripper_adapter':'bounded_projection_v1','maximum_gripper_extrapolation':.10,
            'checkpoint':str(a.checkpoint),'checkpoint_step_label':a.checkpoint.name})
    if a.websocket_compression:
        import dataclasses
        cfg=dataclasses.replace(cfg,policy_metadata={**cfg.policy_metadata,
            'websocket_compression':a.websocket_compression,'transport_pixels':'lossless',
            'jax_policy_seed_per_connection':None if a.evaluation_noise_seeds else 0,
            'evaluation_noise_seed_source':'transport_header_or_default_zero' if a.evaluation_noise_seeds else 'fixed_zero',
            'serialized_evaluation_connections':True})
    if not a.serve:
        print({'status':'configuration_only','model_loaded':False,'action_semantics':'ur5_absolute_joint_goal_v1'})
        return
    if not a.checkpoint.is_dir():raise FileNotFoundError('A local trained checkpoint is required; no automatic download')
    if not a.tokenizer or not a.lerobot_cache:p.error('Serving requires local --tokenizer and --lerobot-cache for real-data warmup')
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from robot_exam.vla.offline_assets import local_assets_only,verify_checkpoint_statistics
    verify_checkpoint_statistics(a.assets/'original_ur5/norm_stats.json',a.checkpoint/'assets/original_ur5/norm_stats.json')
    with local_assets_only(a.tokenizer):
        policy=create_trained_policy(cfg,a.checkpoint)
        # Compile using a real train observation before admitting timed control requests.
        import dataclasses
        import numpy as np
        from openpi.training import data_loader
        factory=dataclasses.replace(cfg.data,dry_check=True)
        dataset=data_loader.create_torch_dataset(factory.create(cfg.assets_dirs,cfg.model),cfg.model.action_horizon,cfg.model)
        sample=dict(dataset[0]);obs={'state':np.asarray(sample['observation.state']),
            'front_rgb':np.asarray(sample['observation.images.front']),'prompt':str(sample['prompt'])}
        if a.wrist_camera:obs['wrist_rgb']=np.asarray(sample['observation.images.wrist'])
        start=time.monotonic();prediction=policy.infer(obs)
        print(json.dumps({'status':'REAL_MODEL_WARMUP_COMPLETE','seconds':time.monotonic()-start,
            'action_shape':list(prediction['actions'].shape),'gripper_projection':prediction.get('gripper_projection'),
            'physical_success':'NOT_EVALUATED'}),flush=True)
        server_type=WebsocketPolicyServer
        if a.websocket_compression:
            class CompressedServer(WebsocketPolicyServer):
                _episode_active = False
                async def _handler(self,websocket):
                    # Serial evaluation: each rollout has its own connection
                    # and the same policy noise seed, independent of preceding
                    # rollout length. Seed/IDs are never model observation input.
                    import jax
                    if self._episode_active:
                        await websocket.close(code=1013,reason='Serial physical evaluation: another episode is active')
                        return
                    self._episode_active=True
                    original_metadata=self._metadata
                    try:
                        seed=evaluation_noise_seed(websocket.request.headers,a.evaluation_noise_seeds)
                        self._policy._rng=jax.random.key(seed)
                        self._metadata={**original_metadata,'episode_policy_noise_seed':seed,
                                        'evaluation_noise_control_enabled':a.evaluation_noise_seeds}
                        await super()._handler(websocket)
                    finally:
                        self._metadata=original_metadata
                        self._episode_active=False
                async def run(self):
                    from websockets.asyncio.server import serve
                    async with serve(self._handler,self._host,self._port,compression='deflate',max_size=None) as server:
                        await server.serve_forever()
            server_type=CompressedServer
        server_type(policy,host='127.0.0.1',port=a.port,metadata=cfg.policy_metadata).serve_forever()


if __name__=='__main__':main()
