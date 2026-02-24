"""I/O utilities for model weights and metadata."""

import os
import json
import datetime


def weights_meta_path(weights_path: str) -> str:
    """Get metadata path for weights file.
    
    Convention: bestG_....weights.h5 -> bestG_....meta.json
    
    Args:
        weights_path: Path to weights file
        
    Returns:
        Path to metadata file
    """
    for ext in (".weights.pkl", ".weights.h5"):
        if weights_path.endswith(ext):
            return weights_path[:-len(ext)] + ".meta.json"
    return weights_path + ".meta.json"


def write_weights_meta(weights_path: str, model_cfg, extra=None) -> str:
    """Write metadata sidecar for weights file.
    
    This makes it harder to accidentally evaluate with the wrong generator/config.
    
    Args:
        weights_path: Path to weights file
        model_cfg: ModelConfig instance
        extra: Extra metadata to include
        
    Returns:
        Path to metadata file
    """
    meta = {
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "generator_type": str(getattr(model_cfg, "generator_type", "")).lower().strip(),
        "z_dim": int(getattr(model_cfg, "z_dim", 0)),
        "kernel": int(getattr(model_cfg, "kernel", 0)),
        "dilations": list(getattr(model_cfg, "dilations", [])),
        "causal": bool(getattr(model_cfg, "causal", True)),
        "use_skip_connections": bool(getattr(model_cfg, "use_skip_connections", True)),
        "g_ch": int(getattr(model_cfg, "g_ch", 0)),
        "g_ch_hidden": int(getattr(model_cfg, "g_ch_hidden", 0)),
    }
    if extra:
        meta.update(extra)

    mp = weights_meta_path(weights_path)
    os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return mp


def assert_weights_compatible(weights_path: str, model_cfg):
    """Assert that weights metadata matches current config.
    
    Args:
        weights_path: Path to weights file
        model_cfg: ModelConfig instance
        
    Raises:
        ValueError: If metadata doesn't match config
    """
    mp = weights_meta_path(weights_path)
    if not os.path.exists(mp):
        return

    with open(mp, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected = str(getattr(model_cfg, "generator_type", "")).lower().strip()
    got = str(meta.get("generator_type", "")).lower().strip()
    if expected and got and expected != got:
        raise ValueError(
            f"Weights/meta generator_type mismatch: config={expected} vs meta={got}. "
            f"Either change generator_type or pick matching weights."
        )

    # Extra guards
    for k in ["z_dim", "kernel"]:
        if k in meta and hasattr(model_cfg, k):
            if int(getattr(model_cfg, k)) != int(meta[k]):
                raise ValueError(
                    f"Weights/meta mismatch for {k}: "
                    f"config={getattr(model_cfg, k)} vs meta={meta[k]}"
                )
