"""Configuration: model + task + method YAML files (with ``inherit:``) and the GPU-memory tier preset.

    python -m rbd.config show   --model qwen35_9b --task code_mt --method rbd
    python -m rbd.config preset --model qwen3_4b [--gpus 0,1]     # shell assignments for scripts/*.sh

Tiers: gpu48 (< 75 GB per GPU), gpu80 (75-120 GB), gpu140 (>= 120 GB); see configs/hardware/presets.yaml.
Every preset value can be overridden with RBD_PLACEMENT, RBD_MEM_UTIL, RBD_MAX_NUM_SEQS, RBD_MICRO_TOKENS,
RBD_TIER. Context length is never reduced by a preset: it always comes from the model config.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = resolve(path)
    cfg = yaml.safe_load(p.read_text()) or {}
    parent = cfg.pop("inherit", None)
    return _merge(load_yaml(p.parent / parent), cfg) if parent else cfg


def load(model: str, task: str, method: str) -> Dict[str, Any]:
    def pick(kind: str, name: str) -> Dict[str, Any]:
        p = Path(name)
        return load_yaml(p if p.suffix == ".yaml" else f"configs/{kind}/{name}.yaml")

    cfg = {"model": pick("models", model), "task": pick("tasks", task), "method": pick("methods", method)}
    cfg["hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return cfg


def model_path(mcfg: Dict[str, Any]) -> str:
    """Local snapshot if RBD_MODEL_PATH / models/<name> exists, else the HF id."""
    env = os.environ.get("RBD_MODEL_PATH")
    if env:
        return env
    local = resolve(f"models/{mcfg['name']}")
    return str(local) if (local / "config.json").exists() else mcfg["hf_id"]


def detect_gpus() -> List[Dict[str, Any]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = [{"index": i.strip(), "mem_gb": float(m) / 1024} for i, m in (l.split(",") for l in out.strip().splitlines())]
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if vis:  # children address GPUs by the visible ids; memory taken from nvidia-smi
        mem = min((g["mem_gb"] for g in gpus), default=0.0)
        return [{"index": v, "mem_gb": mem} for v in vis]
    return gpus


def tier_of(mem_gb: float) -> str:
    return "gpu140" if mem_gb >= 120 else ("gpu80" if mem_gb >= 75 else "gpu48")


def preset(model: str, gpu_ids: Optional[List[str]] = None, mem_gb: Optional[float] = None) -> Dict[str, Any]:
    gpus = detect_gpus()
    ids = gpu_ids or [g["index"] for g in gpus]
    mem = mem_gb if mem_gb is not None else min((g["mem_gb"] for g in gpus), default=0.0)
    if not ids:
        raise RuntimeError("no NVIDIA GPU visible")
    tier = os.environ.get("RBD_TIER") or tier_of(mem)
    p = dict(yaml.safe_load(resolve("configs/hardware/presets.yaml").read_text())[tier][Path(model).stem])
    p["placement"] = os.environ.get("RBD_PLACEMENT", p["placement"])
    if len(ids) < p.get("min_gpus", 1):
        raise RuntimeError(f"{model} on {tier} needs >= {p['min_gpus']} GPUs (split placement), got {len(ids)}")
    if p["placement"] == "split":
        half = max(1, len(ids) // 2)
        server, train = ids[:half], ids[half:]
    else:
        server = train = ids
    return {"tier": tier, "gpu_mem_gb": round(mem, 1), "placement": p["placement"],
            "server_gpus": server, "train_gpus": train,
            "mem_util": float(os.environ.get("RBD_MEM_UTIL", p["mem_util"])),
            "max_num_seqs": int(os.environ.get("RBD_MAX_NUM_SEQS", p["max_num_seqs"])),
            "micro_tokens": int(os.environ.get("RBD_MICRO_TOKENS", p["micro_tokens"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--model", required=True)
    s.add_argument("--task", default="code_mt")
    s.add_argument("--method", default="rbd")
    p = sub.add_parser("preset")
    p.add_argument("--model", required=True)
    p.add_argument("--gpus", default=None)
    p.add_argument("--mem-gb", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "show":
        print(json.dumps(load(a.model, a.task, a.method), indent=2))
        return
    pr = preset(a.model, a.gpus.split(",") if a.gpus else None, a.mem_gb)
    for k, v in pr.items():
        print(f"RBD_{k.upper()}={shlex.quote(','.join(v) if isinstance(v, list) else str(v))}")


if __name__ == "__main__":
    main()
