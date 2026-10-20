#!/usr/bin/env bash
# Read-only environment check for the 8-GPU validation run; installs nothing. Run it inside the Python
# environment you intend to use (an existing vLLM 0.29.0 environment is expected to work as is):
#   bash cluster/remote/check_env.sh            # on a login node: packages, git, unshare, disk
#   bash cluster/remote/check_env.sh --gpu      # on a GPU node: also GPU count / memory, CPU cores, RAM
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-python}"
${PYTHON} - "${1:-}" <<'PY'
import importlib, os, shutil, subprocess, sys
want = {"vllm": "0.29.0", "torch": "2.13.0", "transformers": "5.12.1", "peft": "0.19.1"}
other = ["numpy", "yaml", "aiohttp", "safetensors", "huggingface_hub"]
bad, warn = [], []
print(f"python {sys.version.split()[0]} ({sys.executable})")
for m in list(want) + other:
    try:
        v = getattr(importlib.import_module(m), "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        bad.append(m); print(f"  {m:16s} MISSING ({type(e).__name__})"); continue
    note = ""
    if m in want and v.split("+")[0] != want[m]:
        note = f"   (reported runs used {want[m]})"; warn.append(m)
    print(f"  {m:16s} {v}{note}")
for tool in ("git", "nvidia-smi"):
    print(f"  {tool:16s} {shutil.which(tool) or 'NOT FOUND'}")
    if not shutil.which(tool) and tool == "git":
        bad.append("git")
un = bool(shutil.which("unshare")) and subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0
print(f"  unshare -rn      {'available (agent commands run without network)' if un else 'not available (optional)'}")
st = os.statvfs(".")
print(f"  free disk here   {st.f_bavail * st.f_frsize / 2**30:.0f} GiB (need about 60 GiB)")
if sys.argv[1] == "--gpu":
    import torch
    n = torch.cuda.device_count()
    mems = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n)]
    print(f"  GPUs             {n}: " + ", ".join(f"{torch.cuda.get_device_properties(i).name} {m:.0f} GiB"
                                                  for i, m in enumerate(mems)))
    if n != 8 or min(mems, default=0) < 120:
        bad.append("need 8 GPUs with >= 120 GiB (H200)")
    cpus = len(os.sched_getaffinity(0))
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    print(f"  CPU cores        {cpus} (want >= 64)    host RAM {ram:.0f} GiB (want >= 400)")
    if cpus < 32:
        bad.append("fewer than 32 CPU cores")
print("RESULT:", "FAIL " + ", ".join(bad) if bad else "OK" + (f" (version differences: {', '.join(warn)})" if warn else ""))
sys.exit(1 if bad else 0)
PY
