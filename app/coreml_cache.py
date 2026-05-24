"""CoreML model export and caching for Apple Neural Engine inference.

On Apple Silicon, YOLO models can be exported to CoreML format (.mlpackage)
which runs on the Apple Neural Engine for 5-10x faster inference at lower
power than MPS/GPU. Exported models are cached to avoid re-export on restart.
"""

import hashlib
import os

_SENTINEL = ".export_complete"
_HASH_FILE = ".source_hash"
_HASH_BYTES = 64 * 1024


def _source_hash(model_path):
    """Fast partial hash of the source .pt file for staleness detection."""
    try:
        h = hashlib.md5()
        with open(model_path, "rb") as f:
            h.update(f.read(_HASH_BYTES))
        return h.hexdigest()
    except Exception:
        return None


def _is_cache_valid(cached_dir, source_model_path):
    """Check that a cached CoreML model is complete and not stale."""
    sentinel = os.path.join(cached_dir, _SENTINEL)
    if not os.path.isfile(sentinel):
        return False
    if os.path.isfile(source_model_path):
        hash_file = os.path.join(cached_dir, _HASH_FILE)
        if os.path.isfile(hash_file):
            with open(hash_file) as f:
                cached_hash = f.read().strip()
            current_hash = _source_hash(source_model_path)
            if current_hash and cached_hash != current_hash:
                return False
        else:
            source_mtime = os.path.getmtime(source_model_path)
            cache_mtime = os.path.getmtime(sentinel)
            if source_mtime > cache_mtime:
                return False
    return True


def get_coreml_model_path(model_name, root_dir):
    """Return cached CoreML model path, or export and cache it.

    Returns the path to the cached .mlpackage on success, None on failure.
    Requires coremltools to be installed for the initial export.
    """
    cache_dir = os.path.join(root_dir, ".coreml_cache")
    base = os.path.splitext(os.path.basename(model_name))[0]
    cached = os.path.join(cache_dir, base + "_coreml_model")

    if os.path.isdir(cached) and _is_cache_valid(cached, model_name):
        print(f"  [AI] Using cached CoreML model: {cached}")
        return cached

    if os.path.isdir(cached):
        import shutil
        print(f"  [AI] Removing stale/incomplete CoreML cache: {cached}")
        shutil.rmtree(cached, ignore_errors=True)

    try:
        from ultralytics import YOLO
        import shutil
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  [AI] Exporting {model_name} to CoreML (first-run only, may take 30-60s)...")
        tmp_model = YOLO(model_name)
        exported_path = tmp_model.export(format="coreml", imgsz=640, nms=True)
        if exported_path and os.path.exists(exported_path):
            shutil.move(exported_path, cached)
            sentinel = os.path.join(cached, _SENTINEL)
            with open(sentinel, "w") as f:
                f.write("ok")
            src_hash = _source_hash(model_name)
            if src_hash:
                with open(os.path.join(cached, _HASH_FILE), "w") as f:
                    f.write(src_hash)
            print(f"  [AI] CoreML export cached at: {cached}")
            return cached
    except Exception as e:
        print(f"  [AI] CoreML export failed ({e}), will use standard model")
        if os.path.isdir(cached):
            import shutil
            shutil.rmtree(cached, ignore_errors=True)
    return None
