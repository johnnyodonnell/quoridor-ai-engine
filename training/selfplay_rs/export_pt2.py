"""Export the net to an AOTInductor .pt2 package (fused, Python-free forward).

The orchestrator does this itself at startup; this is a standalone helper for
benchmarks and parity checks. The package bakes in a STATIC batch, which must
match BATCH in pipeline.rs. Run with the training venv from training/selfplay_rs/:
    python export_pt2.py [checkpoint.pt] [out.pt2] [batch]
"""
import sys

import torch

sys.path.insert(0, "..")
from net import (BOARD_SIZE, INPUT_CHANNELS, N_BLOCKS, N_FILTERS,  # noqa: E402
                 SERVING_BATCH, QuoridorNet)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "../../../runs/run1/serving_weights.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "serving_model.pt2"
# Default to the production batch; the override exists for ad-hoc spikes (e.g. a
# parity fixture, or a mismatched .pt2 to test the worker's startup guard).
B = int(sys.argv[3]) if len(sys.argv) > 3 else SERVING_BATCH

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
net = QuoridorNet(n_blocks=cfg.get("n_blocks") or N_BLOCKS,
                  n_filters=cfg.get("n_filters") or N_FILTERS)
net.load_state_dict(ck["weights"])
net.eval().to("cuda").bfloat16()

x = torch.zeros(B, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    ep = torch.export.export(net, (x,))
    path = torch._inductor.aoti_compile_and_package(ep, package_path=OUT)
print(f"wrote {path} (batch={B})")
