#!/usr/bin/env bash
set -Eeuo pipefail
sdaa_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$sdaa_dir/scripts/run-local-examples.sh"
