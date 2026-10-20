#!/usr/bin/env bash
# Build the checker cases (CPU) for one shard of every given problems file: bash scripts/sel_build.sh SHARD NSHARDS F1 [F2 ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SHARD="$1"; NSH="$2"; shift 2
for f in "$@"; do
  ${PYTHON} -m selectrl.checker.build --problems "$f" --cache cache --shard "${SHARD}" --nshards "${NSH}"
done
log "build shard ${SHARD}/${NSH} done"
