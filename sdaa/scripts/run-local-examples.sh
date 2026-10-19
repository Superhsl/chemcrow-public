#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/common.sh"
cd "$SDAA_REPO_ROOT"
for existing in "$SDAA_WORK_DIR/device-smoke.json" "$SDAA_WORK_DIR/device-smoke.log" "$SDAA_WORK_DIR/official-api-command.log" \
                "$SDAA_WORK_DIR/molecular-transformer" "$SDAA_WORK_DIR/official-api"; do
  if [[ -e "$existing" || -L "$existing" ]]; then
    printf 'Existing output would be overwritten: %s. Use a fresh CHEMCROW_WORK_DIR and CHEMCROW_CONTAINER_NAME for a new run.\n' "$existing" >&2
    exit 1
  fi
done
mkdir -p "$SDAA_WORK_DIR/molecular-transformer" "$SDAA_WORK_DIR/official-api"
(cd "$SDAA_MODEL_DIR" && sha256sum -c "$SDAA_REPO_ROOT/sdaa/locks/model-weights.sha256")
sha256sum -c sdaa/locks/wheels.sha256
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$SDAA_CONTAINER_NAME" \
  python -B /workspace/repro/sdaa/scripts/device-smoke.py \
  > "$SDAA_WORK_DIR/device-smoke.log" 2>&1
"$SDAA_REPO_ROOT/sdaa/scripts/run-inference.sh"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$SDAA_CONTAINER_NAME" \
  python -B /workspace/repro/sdaa/scripts/run-official-api.py \
  > "$SDAA_WORK_DIR/official-api-command.log" 2>&1
python3 -B - "$SDAA_WORK_DIR" <<'PY'
import json
import pathlib
import sys
work = pathlib.Path(sys.argv[1])
cli = json.loads((work / "molecular-transformer/result.json").read_text())
api = json.loads((work / "official-api/result.json").read_text())
assert cli["device"] == "sdaa:0"
assert api["http_status"] == 200
assert api["official_status"] == "SUCCESS"
assert api["same_ranked_products_as_cli"]
print("SDAA local examples: PASS")
PY
