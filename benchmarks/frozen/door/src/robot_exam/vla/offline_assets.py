"""Use an explicitly supplied tokenizer; prohibit implicit model/asset downloads."""
from contextlib import contextmanager
from pathlib import Path
import json

TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"


def verify_checkpoint_statistics(expected, restored):
    # Orbax/OpenPI reserializes JSON whitespace; compare content, not file bytes.
    if json.loads(Path(expected).read_text()) != json.loads(Path(restored).read_text()):
        raise ValueError('Checkpoint normalization differs from frozen training-only statistics')


@contextmanager
def local_assets_only(tokenizer_path, download_module=None):
    tokenizer = Path(tokenizer_path).resolve()
    if not tokenizer.is_file() or tokenizer.stat().st_size == 0:
        raise FileNotFoundError("Provide an already downloaded, nonempty PaliGemma tokenizer file")
    if download_module is None:
        from openpi.shared import download as download_module
    original = download_module.maybe_download

    def local(path, *args, **kwargs):
        if str(path) == TOKENIZER_URI:
            return tokenizer
        text = str(path)
        if "://" in text:
            raise RuntimeError(f"Implicit network asset retrieval is disabled: {text}")
        candidate = Path(text).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    download_module.maybe_download = local
    try:
        yield
    finally:
        download_module.maybe_download = original
