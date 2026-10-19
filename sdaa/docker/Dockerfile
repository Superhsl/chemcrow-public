ARG BASE_IMAGE=lasermpnn-sdaa321-loong:torch2.12.0-sdaa3.2.1-py312
FROM ${BASE_IMAGE}

ARG APP_UID=1006
ARG APP_GID=1006
USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/opt/tecoai/lib64

RUN python3 -m venv --system-site-packages /opt/venvs/chemcrow-sdaa-py312 \
    && printf '%s\n' \
        '/opt/venvs/lasermpnn-sdaa321-py312/lib/python3.12/site-packages' \
        '/home/py312/lib/python3.12/site-packages' \
        > /opt/venvs/chemcrow-sdaa-py312/lib/python3.12/site-packages/teco-base.pth \
    && /opt/venvs/chemcrow-sdaa-py312/bin/python -B -c 'import torch, torch_sdaa, rdkit; print(torch.__version__, torch_sdaa.__version__, rdkit.__version__)'


COPY sdaa/vendor /tmp/sdaa/vendor
COPY sdaa/locks/wheels.sha256 /tmp/sdaa/locks/wheels.sha256
RUN cd /tmp \
    && sha256sum -c sdaa/locks/wheels.sha256 \
    && /opt/venvs/chemcrow-sdaa-py312/bin/python -m pip install --no-deps --no-index --no-cache-dir \
        /tmp/sdaa/vendor/configargparse-1.8.0-py3-none-any.whl \
        /tmp/sdaa/vendor/torchtext-0.5.0-py3-none-any.whl \
        /tmp/sdaa/vendor/OpenNMT_py-2.2.0-py3-none-any.whl

RUN if ! getent group "${APP_GID}" >/dev/null; then groupadd --gid "${APP_GID}" chemcrow; fi \
    && if ! getent passwd "${APP_UID}" >/dev/null; then \
         useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/bash chemcrow; \
       fi \
    && mkdir -p /home/chemcrow /workspace/repro/work \
    && chown -R "${APP_UID}:${APP_GID}" /home/chemcrow /workspace/repro/work

ENV VIRTUAL_ENV=/opt/venvs/chemcrow-sdaa-py312 \
    PATH=/opt/venvs/chemcrow-sdaa-py312/bin:/opt/tecoai/bin:${PATH} \
    PYTHONPATH=/workspace/repro/chemcrow-public \
    HOME=/home/chemcrow

WORKDIR /workspace/repro
USER ${APP_UID}:${APP_GID}
CMD ["sleep", "infinity"]
