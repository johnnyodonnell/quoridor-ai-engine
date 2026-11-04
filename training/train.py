"""Training step: pull a minibatch from the replay buffer, compute the
AlphaZero loss (policy cross-entropy + value MSE; L2 via AdamW weight decay),
step the optimizer."""

import numpy as np
import torch
import torch.nn.functional as F

from net import BOARD_SIZE, INPUT_CHANNELS, POLICY_SIZE

STATE_LEN = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE


class ReplayBuffer:
    """Ring buffer of (state, pi, z) rows in preallocated numpy arrays.

    `mirror` is (state_perm, policy_perm) from quoridor_rs; when given, a random
    half of every sampled minibatch is reflected left↔right (an exact symmetry
    of Quoridor), doubling the effective data.
    """

    def __init__(self, capacity, mirror=None):
        self.capacity = capacity
        self.states = np.zeros((capacity, STATE_LEN), dtype=np.float32)
        self.pis = np.zeros((capacity, POLICY_SIZE), dtype=np.float32)
        self.zs = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.head = 0
        self.mirror = None
        if mirror is not None:
            self.mirror = (np.asarray(mirror[0]), np.asarray(mirror[1]))

    def __len__(self):
        return self.size

    def add_many(self, rows):
        for state, pi, z in rows:
            self.states[self.head] = state.reshape(-1)
            self.pis[self.head] = pi
            self.zs[self.head] = z
            self.head = (self.head + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if self.size == 0:
            return None
        idxs = rng.integers(self.size, size=batch_size)
        states = self.states[idxs]
        pis = self.pis[idxs]
        zs = self.zs[idxs]
        if self.mirror is not None:
            flip = rng.random(batch_size) < 0.5
            state_perm, policy_perm = self.mirror
            states[flip] = states[flip][:, state_perm]
            pis[flip] = pis[flip][:, policy_perm]
        states = states.reshape(batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        return states, pis, zs


def train_step(net, opt, batch, device):
    states, pis, zs = batch
    s = torch.from_numpy(states).to(device)
    pi = torch.from_numpy(pis).to(device)
    z = torch.from_numpy(zs).to(device)

    logits, v = net(s)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }
