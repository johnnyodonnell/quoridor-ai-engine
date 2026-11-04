"""AlphaZero-style ResNet for Quoridor.

6 residual blocks × 64 filters over the 9×9 board. Input is the [8, 9, 9]
side-to-move encoding from quoridor_core (board.rs `encode_into`). The policy
head outputs 140 logits (12 relative pawn moves + 64 h-walls + 64 v-walls); the
value head outputs a scalar in [-1, 1] (tanh) from the side-to-move's POV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mirrors quoridor_core/src/board.rs (kept here so torch-only tooling such as
# export.py doesn't need the native extension).
BOARD_SIZE = 9
INPUT_CHANNELS = 8
POLICY_SIZE = 140

N_BLOCKS = 6
N_FILTERS = 64

# GPU batch width the self-play serving_model.pt2 is exported at. The package is
# compiled with a STATIC batch (no dynamic_shapes), so the Rust self-play worker
# must use the same value: it is `BATCH` in selfplay_rs/src/pipeline.rs and is
# enforced at worker startup by aoti_check_batch (shim.cpp). Keep the two in sync.
SERVING_BATCH = 512


class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class QuoridorNet(nn.Module):
    def __init__(self, n_blocks=N_BLOCKS, n_filters=N_FILTERS):
        super().__init__()
        cells = BOARD_SIZE * BOARD_SIZE
        self.stem_conv = nn.Conv2d(INPUT_CHANNELS, n_filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(n_filters)
        self.tower = nn.ModuleList([ResBlock(n_filters) for _ in range(n_blocks)])

        # Policy head: 1×1 conv to 2 channels, then linear to POLICY_SIZE.
        self.policy_conv = nn.Conv2d(n_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * cells, POLICY_SIZE)

        # Value head: 1×1 conv to 1 channel, linear→256→1, tanh.
        self.value_conv = nn.Conv2d(n_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(cells, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.tower:
            h = block(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(-1)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def load_net(path, device):
    """Build a QuoridorNet from a checkpoint/snapshot .pt ({"weights", "config"})."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = QuoridorNet(n_blocks=cfg.get("n_blocks", N_BLOCKS),
                      n_filters=cfg.get("n_filters", N_FILTERS)).to(device)
    net.load_state_dict(ckpt["weights"])
    net.eval()
    return net
