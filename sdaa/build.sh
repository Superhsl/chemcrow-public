#!/usr/bin/env bash
set -Eeuo pipefail
sdaa_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ "$(uname -m)" != loongarch64 ]]; then
  printf 'This SDAA build requires a loongarch64 host.\n' >&2
  exit 1
fi
"$sdaa_dir/scripts/download-model-weights.sh"
exec "$sdaa_dir/scripts/build-and-start.sh"
