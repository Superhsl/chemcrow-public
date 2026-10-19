#!/usr/bin/env bash
SDAA_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SDAA_MODEL_DIR="${CHEMCROW_MODEL_DIR:-$SDAA_REPO_ROOT/../models/molecular-transformer}"
SDAA_WORK_DIR="${CHEMCROW_WORK_DIR:-$SDAA_REPO_ROOT/../work/sdaa-repo}"
SDAA_CONTAINER_NAME="${CHEMCROW_CONTAINER_NAME:-chemcrow-sdaa-repo}"
SDAA_IMAGE_TAG="${CHEMCROW_IMAGE_TAG:-chemcrow-sdaa:repo-torch2.12-sdaa3.2.1-py312}"
if [[ "$SDAA_MODEL_DIR" != /* || "$SDAA_WORK_DIR" != /* ]]; then
  printf 'CHEMCROW_MODEL_DIR and CHEMCROW_WORK_DIR must be absolute paths.\n' >&2
  return 1
fi
