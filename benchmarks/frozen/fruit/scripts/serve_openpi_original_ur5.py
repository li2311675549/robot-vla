#!/usr/bin/env python3
"""Prepared inference server; requires a real user-provided trained checkpoint."""
import argparse
from pathlib import Path
import sys
from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--serve',action='store_true')
    p.add_argument('--repo-id',default='local/robot_exam_schema3_train');p.add_argument('--wrist-camera',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.openpi_root/'src'))
    from robot_exam.vla.openpi_config import make_config
    cfg=make_config(a.repo_id,a.assets,a.checkpoint,ROOT/'checkpoints',dry_check=not a.serve,wrist_camera=a.wrist_camera)
    if not a.serve:
        print({'status':'configuration_only','model_loaded':False,'action_semantics':'ur5_absolute_joint_goal_v1'})
        return
    if not a.checkpoint.is_dir():raise FileNotFoundError('A local trained checkpoint is required; no automatic download')
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=create_trained_policy(cfg,a.checkpoint)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=cfg.policy_metadata).serve_forever()


if __name__=='__main__':main()
