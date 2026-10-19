#!/usr/bin/env python3
"""Run the official Molecular Transformer CLI via the Torch-SDAA CUDA bridge."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch_sdaa


ROOT = Path("/workspace/repro")
SOURCE_APP = (
    ROOT
    / "chemcrow-public/chemcrow/docker/molecular-transformer/app.py"
)
MODEL = ROOT / "models/molecular-transformer/USPTO480k_model_step_400000.pt"
WORK = ROOT / "work/molecular-transformer"
SMILES = "CCOCCCCO.CC(=O)Cl"
MODEL_SHA256 = "1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    model_digest = sha256_file(MODEL)
    if model_digest != MODEL_SHA256:
        raise RuntimeError(f"Unexpected checkpoint SHA-256: {model_digest}")
    if not torch.sdaa.is_available():
        raise RuntimeError("SDAA is not available in the ChemCrow container")

    sys.path.insert(0, str(SOURCE_APP.parent))
    import app as official_app  # noqa: PLC0415

    WORK.mkdir(parents=True, exist_ok=True)
    source_path = WORK / "input.txt"
    output_path = WORK / "output.txt"
    log_path = WORK / "inference.log"
    result_path = WORK / "result.json"
    source_path.write_text(official_app.smiles_tokenizer(SMILES), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "onmt.bin.translate",
        "--model",
        str(MODEL),
        "--src",
        str(source_path),
        "--output",
        str(output_path),
        "--n_best",
        "5",
        "--beam_size",
        "10",
        "--max_length",
        "300",
        "--batch_size",
        "1",
        "--gpu",
        "0",
    ]
    child_env = os.environ.copy()
    child_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    child_env["TORCH_SDAA_AUTOLOAD"] = "cuda_migrate"
    completed = subprocess.run(
        command,
        cwd=WORK,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)

    predictions = output_path.read_text(encoding="utf-8").splitlines()
    result = {
        "smiles": SMILES,
        "tokenized_input": source_path.read_text(encoding="utf-8"),
        "products": predictions,
        "device": str(torch.device("sdaa:0")),
        "accelerator": torch.sdaa.get_device_name(0),
        "torch_sdaa": torch_sdaa.__version__,
        "torch": torch.__version__,
        "sdaa_device_count": torch.sdaa.device_count(),
        "model_sha256": model_digest,
        "command": command,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
