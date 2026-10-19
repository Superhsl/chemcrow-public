#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/common.sh"
for name in input.txt output.txt inference.log result.json command.log; do
  existing="$SDAA_WORK_DIR/molecular-transformer/$name"
  if [[ -e "$existing" || -L "$existing" ]]; then
    printf 'Existing inference output would be overwritten: %s. Use a fresh output directory and container.\n' "$existing" >&2
    exit 1
  fi
done
mkdir -p "$SDAA_WORK_DIR/molecular-transformer"
(cd "$SDAA_MODEL_DIR" && sha256sum -c "$SDAA_REPO_ROOT/sdaa/locks/model-weights.sha256")
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$SDAA_CONTAINER_NAME" \
  python -B /workspace/repro/sdaa/scripts/run-molecular-transformer.py \
  | tee "$SDAA_WORK_DIR/molecular-transformer/command.log"
