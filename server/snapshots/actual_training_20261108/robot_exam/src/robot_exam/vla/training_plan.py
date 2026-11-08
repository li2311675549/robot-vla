"""Validated, serializable settings for the first server-only LoRA experiment."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class TrainingPlan:
    run_name: str = "dual_fruit_pilot_v1"
    steps: int = 1000
    batch_size: int = 2
    warmup_steps: int = 100
    peak_lr: float = 2.5e-5
    end_lr: float = 2.5e-6
    save_interval: int = 250
    keep_period: int = 250
    log_interval: int = 10
    seed: int = 42
    num_workers: int = 0
    validation_repo_id: str = "local/robot_exam_dual_release_validation"
    validation_frames_per_episode: int = 3
    validation_seed: int = 731

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", self.run_name):
            raise ValueError("run_name must be a safe single directory component")
        for name in ("steps", "batch_size", "save_interval", "keep_period", "log_interval",
                     "validation_frames_per_episode"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "num_workers", "seed", "validation_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup must finish before the final training step")
        if not all(math.isfinite(x) and x > 0 for x in (self.peak_lr, self.end_lr)):
            raise ValueError("Learning rates must be finite and positive")
        if self.end_lr > self.peak_lr:
            raise ValueError("end_lr cannot exceed peak_lr")
        if not self.validation_repo_id.startswith("local/") or ".." in self.validation_repo_id:
            raise ValueError("Use a local validation dataset")
        if not 1 <= self.validation_frames_per_episode <= 5:
            raise ValueError("Use 1..5 fixed validation frames per episode")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)


def episode_sample_indices(episodes, frames_per_episode=1):
    """Equal episode weighting; no selection by outcome or predicted error."""
    indices = []
    offset = 0
    for episode in episodes:
        length = int(episode["length"])
        if length < frames_per_episode:
            raise ValueError("Episode too short for the declared sampling protocol")
        for i in range(frames_per_episode):
            indices.append(offset + min(length - 1, int(length * (i + .5) / frames_per_episode)))
        offset += length
    return indices
