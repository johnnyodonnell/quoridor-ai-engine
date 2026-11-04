"""Forward-parity fixture generator + driver (port of chess-ai-engine's
selfplay_rs/test_forward_parity.py).

Plays random games with quoridor_rs to collect real, non-terminal positions,
saves them with PyTorch fp32, bf16 and torch.compile-bf16 reference outputs plus
per-position legal masks, and writes the checkpoint as weights.safetensors.
With --run it then runs `selfplay_rs forward-check .`, which asserts:
  - Rust BN-folded vs unfolded agree in fp32 (fold math);
  - Rust fp32 explicit-BN forward vs PyTorch fp32: tight numeric match;
  - Rust bf16 vs PyTorch bf16 priors/values within bf16 tolerance.
`selfplay_rs aoti-parity` then checks the AOTI package against the
torch.compile reference (the previous production forward).

Run from training/selfplay_rs/ with the training venv:
    python test_forward_parity.py [checkpoint.pt] [n_positions] [--run]
"""
import glob
import os
import random
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, "..")  # net.py, quoridor_rs
import quoridor_rs  # noqa: E402
from net import N_BLOCKS, N_FILTERS, POLICY_SIZE, QuoridorNet  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
CKPT = args[0] if args else "../../../runs/run1/serving_weights.pt"
N = int(args[1]) if len(args) > 1 else 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck.get("config", {}) or {}
net = QuoridorNet(n_blocks=cfg.get("n_blocks") or N_BLOCKS,
                  n_filters=cfg.get("n_filters") or N_FILTERS)
net.load_state_dict(ck["weights"])
net.eval()
save_file({k: v.contiguous() for k, v in net.state_dict().items()}, "weights.safetensors")

# Real positions: random playouts to varied depths, skipping finished games.
rng = random.Random(0)
positions, masks = [], []
while len(positions) < N:
    b = quoridor_rs.Board()
    for _ in range(rng.randint(0, 60)):
        if b.outcome() is not None:
            break
        b.push(rng.choice(b.legal_moves()))
    if b.outcome() is not None:
        continue
    positions.append(b.encode())
    m = np.zeros(POLICY_SIZE, dtype=np.float32)
    m[b.legal_moves()] = 1.0
    masks.append(m)

x = torch.from_numpy(np.stack(positions)).float()
mask = torch.from_numpy(np.stack(masks)).float()

with torch.no_grad():
    net_f32 = net.to(DEV)
    lf, vf = net_f32(x.to(DEV))
    net_bf16 = net.to(DEV).bfloat16()
    xb = x.to(DEV).bfloat16()
    lb, vb = net_bf16(xb)
    # The previous production forward was torch.compile (Inductor-fused, bf16):
    # the true equivalence target for the (also Inductor-fused) AOTI package.
    cnet = torch.compile(net_bf16)
    lc, vc = cnet(xb)

save_file(
    {
        "input": x.contiguous(),
        "legal_mask": mask.contiguous(),
        "ref_logits_f32": lf.float().cpu().contiguous(),
        "ref_values_f32": vf.float().cpu().contiguous(),
        "ref_logits_bf16": lb.float().cpu().contiguous(),
        "ref_values_bf16": vb.float().cpu().contiguous(),
        "ref_logits_compile": lc.float().cpu().contiguous(),
        "ref_values_compile": vc.float().cpu().contiguous(),
    },
    "fixture.safetensors",
)
print(f"wrote weights.safetensors + fixture.safetensors ({N} real positions)")

if "--run" in sys.argv:
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    sp = os.path.dirname(os.path.dirname(torch.__file__))
    libs = [torch_lib] + sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    env = {**os.environ,
           "LD_LIBRARY_PATH": ":".join(libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    rc = subprocess.run(["./target/release/selfplay_rs", "forward-check", "."], env=env).returncode
    sys.exit(rc)
