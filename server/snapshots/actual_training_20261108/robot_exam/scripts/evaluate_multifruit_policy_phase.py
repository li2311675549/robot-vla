#!/usr/bin/env python3
"""Real two-view policy grasp/place in clutter; replay/mock explicitly labelled."""
import argparse
from pathlib import Path
import sys
import json

from _bootstrap import ROOT
from robot_exam.vla.multifruit_evaluation import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', choices=('openpi', 'replay', 'mock_hold'), required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--expected-checkpoint-label')
    p.add_argument('--expected-normalization-sha256')
    p.add_argument('--policy-noise-seed', type=int, default=0)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--record', action='store_true')
    p.add_argument('--client-source', type=Path, default=Path('/home/li/vla/openpi/packages/openpi-client/src'))
    args = p.parse_args()
    if args.policy == 'openpi' and args.expected_checkpoint_label is None:
        p.error('Pin the checkpoint label before any real policy action')
    sys.path.insert(0, str(args.client_source))

    class LazyRemote:
        client = None

        def get_server_metadata(self):
            from robot_exam.vla.client import connect_openpi_policy
            self.client = connect_openpi_policy(args.host, args.port, compression='deflate',
                                                noise_seed=args.policy_noise_seed)
            return self.client.get_server_metadata()

        def infer(self, observation):
            return self.client.infer(observation)

    remote = LazyRemote() if args.policy == 'openpi' else None
    try:
        result = evaluate(args.fixture, args.output, args.policy, policy=remote,
                          checkpoint=args.expected_checkpoint_label, seconds=args.seconds, record=args.record,
                          expected_norm=args.expected_normalization_sha256)
    finally:
        if remote and remote.client:
            remote.client._ws.close()
    print(json.dumps(result, indent=2))
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
