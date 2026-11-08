"""Held-out validation at saves in the pinned JAX trainer (server-only model work)."""
import dataclasses
import json
from pathlib import Path
import shutil
import time

import numpy as np

from robot_exam.vla.training_plan import episode_sample_indices


class ValidationMonitor:
    """Fixed validation frames/noise; train-only normalization; no optimizer update."""

    def __init__(self, cfg, plan, cache, directory):
        import jax
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.training import data_loader

        self.jax = jax
        self.nnx = nnx
        self.plan = plan
        self.directory = Path(directory)
        meta = Path(cache) / plan.validation_repo_id / "meta/episodes.jsonl"
        episodes = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
        indices = episode_sample_indices(episodes, plan.validation_frames_per_episode)
        val_factory = dataclasses.replace(cfg.data, repo_id=plan.validation_repo_id)
        data = val_factory.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
        dataset = data_loader.transform_dataset(dataset, data)
        self.dataset = dataset
        self.model_module = model_module
        self.indices = indices
        self.eval_fn = jax.jit(self._loss)
        self.best = float("inf")
        path = self.directory / "best_validation.json"
        if path.exists():
            self.best = float(json.loads(path.read_text())["flow_matching_loss"])

    def _loss(self, graphdef, params, rng, observation, actions):
        model = self.nnx.merge(graphdef, params)
        model.eval()
        return self.jax.numpy.mean(model.compute_loss(rng, observation, actions, train=False))

    def __call__(self, train_state, checkpoint_step):
        # Same stochastic noise schedule at each checkpoint; augmentation disabled via train=False.
        start = time.monotonic()
        losses = []
        # Stream batch-one samples: no full-resolution image corpus retained in GPU/host memory.
        for i, index in enumerate(self.indices):
            from openpi import transforms
            item = transforms.ResizeImages(224, 224)(self.dataset[index])
            item = self.jax.tree.map(lambda x: np.asarray(x)[None], item)
            observation = self.model_module.Observation.from_dict(item)
            actions = item['actions']
            rng = self.jax.random.fold_in(self.jax.random.key(self.plan.validation_seed), i)
            loss = self.eval_fn(train_state.model_def, train_state.params, rng, observation, actions)
            losses.append(float(self.jax.device_get(loss)))
        if not np.isfinite(losses).all():
            raise FloatingPointError("Nonfinite held-out flow-matching loss")
        row = {"checkpoint_directory_step": int(checkpoint_step),
               "optimizer_step": int(train_state.step), "repo_id": self.plan.validation_repo_id,
               "flow_matching_loss": float(np.mean(losses)), "sample_count": len(losses),
               "frame_indices": self.indices, "seed": self.plan.validation_seed,
               "wall_seconds": time.monotonic() - start,
               "scope": "offline held-out flow loss, not physical task success"}
        with (self.directory / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["flow_matching_loss"] < self.best:
            self.best = row["flow_matching_loss"]
            (self.directory / "best_validation.json").write_text(json.dumps(row, indent=2) + "\n")
        return row


def run_pinned_training(module, cfg, plan, cache, directory):
    """Install process-local logging/validation hooks without editing upstream code."""
    import jax
    import numpy as np

    directory = Path(directory)
    monitor = ValidationMonitor(cfg, plan, cache, directory)
    original_save = module._checkpoints.save_state
    original_step = module.train_step
    original_init = module.init_train_state if hasattr(module, 'init_train_state') else None

    def record_step(step, loss, gradient):
        row = {"optimizer_step": int(step), "loss": float(loss), "grad_norm": float(gradient),
               "wall_monotonic": time.monotonic()}
        if not np.isfinite([row["loss"], row["grad_norm"]]).all():
            raise FloatingPointError("Nonfinite training loss or gradient")
        with (directory / "train_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def checked_step(*args, **kwargs):
        state, info = original_step(*args, **kwargs)
        jax.debug.callback(record_step, state.step, info["loss"], info["grad_norm"], ordered=True)
        return state, info

    def save_and_validate(manager, state, loader, step):
        if shutil.disk_usage(directory).free < 25 * 1024**3:
            raise RuntimeError('Stop before checkpoint save: less than 25GiB free')
        result = original_save(manager, state, loader, step)
        manager.wait_until_finished()
        monitor(state, step)
        return result

    module.train_step = checked_step
    module._checkpoints.save_state = save_and_validate
    if original_init is not None:
        def audited_init(*args, **kwargs):
            state, sharding = original_init(*args, **kwargs)
            selected = state.params.filter(cfg.trainable_filter)
            rows = []
            for path, value in jax.tree_util.tree_flatten_with_path(selected)[0]:
                rows.append({'path': jax.tree_util.keystr(path), 'shape': list(value.shape),
                             'elements': int(np.prod(value.shape))})
            if not rows or any('llm' in row['path'] and 'lora' not in row['path'] for row in rows):
                raise RuntimeError('Unexpected trainable LLM backbone; check pinned LoRA freeze filter')
            (directory/'trainable_parameters.json').write_text(json.dumps({
                'elements': sum(x['elements'] for x in rows), 'parameters': rows,
                'note': 'Pinned upstream LoRA freeze filter; non-LLM projections/vision may also be trainable'}, indent=2)+'\n')
            return state, sharding
        module.init_train_state = audited_init
    try:
        module.main(cfg)
        jax.effects_barrier()
    finally:
        module.train_step = original_step
        module._checkpoints.save_state = original_save
        if original_init is not None:
            module.init_train_state = original_init
