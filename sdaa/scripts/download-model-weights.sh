#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/common.sh"
model_name=USPTO480k_model_step_400000.pt
file_id=1ywJCJHunoPTB5wr6KdZ8aLv7tMFMBHNy
expected_sha256=1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587
mkdir -p "$SDAA_MODEL_DIR"
model_path="$SDAA_MODEL_DIR/$model_name"
if [[ -e "$model_path" || -L "$model_path" ]]; then
  if [[ -f "$model_path" ]] && [[ "$(sha256sum "$model_path" | cut -d ' ' -f 1)" == "$expected_sha256" ]]; then
    printf 'Official checkpoint already verified: %s\n' "$model_path"
    exit 0
  fi
  printf 'Existing checkpoint differs from the locked SHA-256: %s. Preserve or move it aside before retrying.\n' "$model_path" >&2
  exit 1
fi
download_path="$(mktemp "$SDAA_MODEL_DIR/.$model_name.download.XXXXXX")"
cookie_jar="$(mktemp)"
trap 'rm -f "$download_path" "$cookie_jar"' EXIT
curl --fail --location --silent --show-error --max-time 900 \
  --cookie "$cookie_jar" --cookie-jar "$cookie_jar" \
  --output "$download_path" \
  "https://drive.google.com/uc?export=download&id=$file_id"
if file "$download_path" | grep -qi 'HTML document'; then
  uuid="$(python3 - "$download_path" <<'PY'
import re
import sys
from pathlib import Path
html = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'name="uuid" value="([^"]+)"', html)
if not match:
    raise SystemExit("Google Drive response has no confirmation token")
print(match.group(1))
PY
  )"
  curl --fail --location --silent --show-error --retry 2 --max-time 900 \
    --cookie "$cookie_jar" --cookie-jar "$cookie_jar" \
    --output "$download_path" \
    "https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=t&uuid=$uuid"
fi
if ! file "$download_path" | grep -q 'Zip archive'; then
  printf 'Downloaded file is not a PyTorch ZIP checkpoint.\n' >&2
  exit 1
fi
actual_sha256="$(sha256sum "$download_path" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  printf 'Checkpoint SHA-256 mismatch: %s\n' "$actual_sha256" >&2
  exit 1
fi
if [[ -e "$model_path" || -L "$model_path" ]]; then
  printf 'Checkpoint appeared during download; preserving it: %s\n' "$model_path" >&2
  exit 1
fi
mv -n "$download_path" "$model_path"
(cd "$SDAA_MODEL_DIR" && sha256sum -c "$SDAA_REPO_ROOT/sdaa/locks/model-weights.sha256")
