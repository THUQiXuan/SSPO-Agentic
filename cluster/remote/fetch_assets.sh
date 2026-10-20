#!/usr/bin/env bash
# Download the public inputs of the validation run into the repository root and verify them (needs internet; run on
# a login node if compute nodes are offline):
#   * GitHub release v1.0 of THUQiXuan/SSPO-Agentic: problem files, checker cases, step-30 adapters, reference rows;
#   * Qwen/Qwen3-4B at the pinned revision -> models/qwen3_4b (skipped if RBD_MODEL_PATH points to a snapshot).
# Every archive is checked against cluster/remote/assets.sha256 before it is extracted.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-python}"
URL="${SSPO_RELEASE_URL:-https://github.com/THUQiXuan/SSPO-Agentic/releases/download/v1.0}"
mkdir -p downloads
grep -q "TO_BE_FILLED" cluster/remote/assets.sha256 && { echo "release assets are not published yet" >&2; exit 2; }
while read -r sum name; do
  [[ -z "${name}" || "${sum}" == \#* ]] && continue
  if [[ ! -f downloads/${name} ]]; then
    curl -fL --retry 5 -o downloads/${name}.part "${URL}/${name}"
    mv -f downloads/${name}.part downloads/${name}
  fi
  echo "${sum}  downloads/${name}" | sha256sum -c - || { echo "checksum mismatch: ${name}" >&2; exit 1; }
  tar -xzf downloads/${name}
done < cluster/remote/assets.sha256
if [[ -z "${RBD_MODEL_PATH:-}" && ! -f models/qwen3_4b/config.json ]]; then
  ${PYTHON} - <<'PY'
from huggingface_hub import snapshot_download
from rbd.config import load_yaml
m = load_yaml("configs/models/qwen3_4b.yaml")
snapshot_download(m["hf_id"], revision=m["revision"], local_dir="models/qwen3_4b",
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
PY
fi
${PYTHON} - <<'PY'
import gzip, json
from pathlib import Path
n = {s: sum(1 for _ in gzip.open(f"data/sel/{s}.jsonl.gz", "rt")) for s in ("train", "dev", "test")}
pids = [json.loads(l)["pid"] for s in n for l in gzip.open(f"data/sel/{s}.jsonl.gz", "rt")]
miss = [p for p in pids if not Path(f"cache/checkers/cases/{p}.json.gz").exists()]
print("problems", n, "| checker cases missing:", len(miss))
assert n == {"train": 231, "dev": 65, "test": 128} and not miss
for a in ("sel_sspo_s0", "sel_drgrpo_s0", "sel_pkpo_s0"):
    assert Path(f"reference/adapters/{a}/adapter_config.json").exists(), a
print("assets OK")
PY
