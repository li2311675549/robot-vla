"""Explicit actuator-space projection for bounded learned gripper errors."""
from dataclasses import dataclass

import numpy as np

from robot_exam.vla.interface import validate_actions


@dataclass(frozen=True)
class ProjectGripperOutputs:
    # Ten percent of normalized travel; larger extrapolation is rejected.
    maximum_extrapolation: float = .10

    def __call__(self, data):
        raw = np.asarray(data['actions'])
        if raw.ndim != 2 or raw.shape[1] < 7 or not np.isfinite(raw).all():
            raise ValueError('Nonfinite or malformed learned action chunk')
        actions = raw[:, :7].astype(np.float64, copy=True)
        grip = actions[:, 6].copy()
        excess = np.maximum(-grip, np.maximum(grip - 1., 0.))
        if not 0 <= self.maximum_extrapolation <= .10:
            raise ValueError('Invalid inference projection margin')
        if excess.max() > self.maximum_extrapolation:
            raise ValueError(f'Excessive gripper extrapolation: min={grip.min()}, max={grip.max()}')
        actions[:, 6] = np.clip(grip, 0., 1.)
        return {'actions': validate_actions(actions).astype(np.float32),
                'gripper_projection': {'raw_min': float(grip.min()), 'raw_max': float(grip.max()),
                    'raw_values': grip.tolist(), 'projected_values': int(np.count_nonzero(excess)),
                    'maximum_excess': float(excess.max()), 'margin': self.maximum_extrapolation}}
