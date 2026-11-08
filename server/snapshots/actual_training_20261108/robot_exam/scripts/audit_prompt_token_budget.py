#!/usr/bin/env python3
"""Use the installed native tokenizer class without importing JAX/model code.

Loads only the PaligemmaTokenizer AST node from the pinned local OpenPI source.
Its download hook is replaced by a read-only local asset resolver. No network,
policy, model or optimizer is instantiated.
"""
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve token budget audits')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sentencepiece
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source = a.root / 'openpi/src/openpi/models/tokenizer.py'
    config = a.root / 'openpi/src/openpi/training/config.py'
    model_config = a.root / 'openpi/src/openpi/models/pi0_config.py'
    text = config.read_text().split('case _model.ModelType.PI05:', 1)[1].split('case ', 1)[0]
    assert text.index('_transforms.TokenizePrompt') < text.index('_transforms.PadStatesAndActions')
    assert '200 if self.pi05 else 48' in model_config.read_text()
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'PaligemmaTokenizer')

    def local_asset(url, **kwargs):
        if url != 'gs://big_vision/paligemma_tokenizer.model':
            raise ValueError('Unexpected tokenizer asset requested')
        return a.tokenizer

    namespace = {'np': np, 'sentencepiece': sentencepiece, 'logging': logging,
                 'download': SimpleNamespace(maybe_download=local_asset)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    tokenizer = namespace['PaligemmaTokenizer'](max_len=4096)
    # The larger QA-only limit exposes the native UNTRUNCATED token count;
    # it does not alter the training/model limit of200.
    base = a.root / 'data_preparation'
    old = base / 'green_target_v2'
    merged = base / 'green_timing_merged_v3'
    matched = base / 'executor_matched6_v4'
    profiles = {
        'original173_oldnorm': ([old / 'full' / s for s in ('train', 'validation', 'test')],
                                old / 'manifest/assets/original_ur5/norm_stats.json'),
        'merged185_new127train_norm': ([merged / 'full' / s for s in ('train', 'validation', 'test')],
                                      merged / 'manifest/assets/original_ur5/norm_stats.json'),
        'matched6_frozen121train_norm': ([matched / 'train'],
                                        matched / 'fit_manifest/assets/original_ur5/norm_stats.json'),
    }
    results = {}
    for name, (datasets, norm_file) in profiles.items():
        norm = json.loads(norm_file.read_text())['norm_stats']['state']
        q01, q99 = np.asarray(norm['q01']), np.asarray(norm['q99'])
        report = {'episodes': 0, 'frames': 0, 'legacy': {'minimum': 4096, 'maximum': 0, 'truncated': 0},
                  'green_explicit': {'minimum': 4096, 'maximum': 0, 'truncated': 0},
                  'state_tokenization_dimensions': 7}
        for dataset in datasets:
            info = json.loads((dataset / 'meta/info.json').read_text())
            tasks = {v['task_index']: v['task'] for v in map(json.loads,
                     (dataset / 'meta/tasks.jsonl').read_text().splitlines())}
            for episode in map(json.loads, (dataset / 'meta/episodes.jsonl').read_text().splitlines()):
                index = episode['episode_index']
                file = dataset / info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
                table = pq.ParquetFile(file).read(columns=['observation.state', 'task_index'], use_threads=False)
                states = np.asarray(table['observation.state'].to_pylist())
                normalized = (states - q01) / (q99 - q01 + 1e-6) * 2 - 1
                assert normalized.shape == (len(table), 7)
                for state, task_id in zip(normalized, table['task_index'].to_pylist(), strict=True):
                    skill = tasks[task_id].split('Pick up the ', 1)[1].split(' and ', 1)[0]
                    for style, prompt in (
                        ('legacy', f'Pick up the {skill} and place it at the target.'),
                        ('green_explicit', f'Pick up the {skill} and place it on the green circular target marker on the table.'),
                    ):
                        tokens, mask = tokenizer.tokenize(prompt, state)
                        count = int(mask.sum())
                        assert len(tokens) == 4096
                        report[style]['minimum'] = min(report[style]['minimum'], count)
                        report[style]['maximum'] = max(report[style]['maximum'], count)
                        report[style]['truncated'] += int(count > 200)
                report['episodes'] += 1
                report['frames'] += len(table)
        results[name] = report
    passed = all(r[style]['truncated'] == 0 for r in results.values() for style in ('legacy', 'green_explicit'))
    output = {'status': 'PASS_NO_PROMPT_STATE_TRUNCATION' if passed else 'FAIL_TOKEN_BUDGET',
              'model_limit': 200, 'qa_untruncated_limit': 4096, 'profiles': results,
              'native_tokenizer_ast_sha256': hashlib.sha256(ast.dump(node).encode()).hexdigest(),
              'tokenizer_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'tokenizer_asset_sha256': hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
              'torch_imported': 'torch' in sys.modules, 'jax_imported': 'jax' in sys.modules,
              'model_loaded': False, 'training_started': False, 'network_access': False,
              'scope': 'Read-only native language/state token budget; not model inference or physical success'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
