#!/usr/bin/env python3
"""Exercise the unmodified official Flask route through the SDAA translator bridge."""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/repro")
WORK = ROOT / "work/official-api"
MODEL = ROOT / "models/molecular-transformer/USPTO480k_model_step_400000.pt"
SMILES = "CCOCCCCO.CC(=O)Cl"

WORK.mkdir(parents=True, exist_ok=True)
(WORK / "models").mkdir(exist_ok=True)
link = WORK / "models/USPTO480k_model_step_400000.pt"
if not link.exists():
    link.symlink_to(MODEL)
if link.resolve() != MODEL:
    raise RuntimeError(f"Unexpected model link: {link}")
os.environ["PATH"] = f"{ROOT / 'sdaa/bin'}:{os.environ['PATH']}"
os.environ["ONMT_BRIDGE_LOG"] = str(WORK / "bridge.log")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.chdir(WORK)
sys.path.insert(0, str(ROOT / "chemcrow-public/chemcrow/docker/molecular-transformer"))
import app as official_app

response = official_app.app.test_client().post("/api/v1/run", json={"smiles": SMILES})
payload = response.get_json()
cli = (ROOT / "work/molecular-transformer/output.txt").read_text(encoding="utf-8").splitlines()
expected = [product.replace(" ", "") for product in cli]
received = [product for product in payload["product"] if product]
result = {
    "http_status": response.status_code,
    "official_status": payload["status"],
    "smiles": SMILES,
    "products": received,
    "same_ranked_products_as_cli": received == expected,
    "bridge_log": str(WORK / "bridge.log"),
}
(WORK / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
if response.status_code != 200 or payload["status"] != "SUCCESS" or received != expected:
    raise SystemExit("Official API inference mismatch")
