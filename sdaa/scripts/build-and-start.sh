#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "$0")/common.sh"
cd "$SDAA_REPO_ROOT"
if docker container inspect "$SDAA_CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container %s already exists; inspect or start it before rebuilding.\n' "$SDAA_CONTAINER_NAME" >&2
  exit 1
fi
if [[ ! -d "$SDAA_MODEL_DIR" ]]; then
  printf 'Missing model directory: %s. Run sdaa/scripts/download-model-weights.sh first.\n' "$SDAA_MODEL_DIR" >&2
  exit 1
fi
(cd "$SDAA_MODEL_DIR" && sha256sum -c "$SDAA_REPO_ROOT/sdaa/locks/model-weights.sha256")
sha256sum -c sdaa/locks/wheels.sha256
base_image="$(sed -n 's/^image=//p' sdaa/locks/base-image.lock)"
expected_id="$(sed -n 's/^image_id=//p' sdaa/locks/base-image.lock)"
if [[ -z "$base_image" || -z "$expected_id" ]]; then
  printf 'Incomplete base image lock.\n' >&2
  exit 1
fi
observed_id="$(docker image inspect "$base_image" --format '{{.Id}}')"
if [[ "$observed_id" != "$expected_id" ]]; then
  printf 'Base image ID mismatch: expected %s, found %s\n' "$expected_id" "$observed_id" >&2
  exit 1
fi
host_uid="$(id -u)"
host_gid="$(id -g)"
mkdir -p "$SDAA_WORK_DIR"
docker build \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "APP_UID=$host_uid" \
  --build-arg "APP_GID=$host_gid" \
  -f sdaa/docker/Dockerfile \
  -t "$SDAA_IMAGE_TAG" .
device_args=()
for path in /dev/tcaicard*; do
  if [[ -c "$path" ]]; then
    device_args+=(--device "$path:$path:rwm")
  fi
done
if (( ${#device_args[@]} == 0 )); then
  printf 'No SDAA character devices found.\n' >&2
  exit 1
fi
docker run -d \
  --name "$SDAA_CONTAINER_NAME" \
  --init \
  --user "$host_uid:$host_gid" \
  --shm-size 2g \
  -e HOME=/home/chemcrow \
  -e LD_LIBRARY_PATH=/opt/tecoai/lib64 \
  -e PYTHONPATH=/workspace/repro/chemcrow-public \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$SDAA_REPO_ROOT:/workspace/repro/chemcrow-public:ro" \
  -v "$SDAA_MODEL_DIR:/workspace/repro/models/molecular-transformer:ro" \
  -v "$SDAA_REPO_ROOT/sdaa:/workspace/repro/sdaa:ro" \
  -v "$SDAA_WORK_DIR:/workspace/repro/work:rw" \
  "${device_args[@]}" \
  "$SDAA_IMAGE_TAG" sleep infinity
docker inspect "$SDAA_CONTAINER_NAME" --format 'status={{.State.Status}} image={{.Image}} user={{.Config.User}} devices={{len .HostConfig.Devices}}'
