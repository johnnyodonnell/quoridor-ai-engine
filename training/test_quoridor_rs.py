"""Tests for the quoridor_rs bindings and the Python training glue.
Run from training/:  python -m pytest -q"""

import numpy as np
import pytest

import quoridor_rs
from net import BOARD_SIZE, INPUT_CHANNELS, POLICY_SIZE
from train import ReplayBuffer


def test_constants_match_net():
    assert quoridor_rs.INPUT_CHANNELS == INPUT_CHANNELS
    assert quoridor_rs.BOARD_SIZE == BOARD_SIZE
    assert quoridor_rs.POLICY_SIZE == POLICY_SIZE


def test_board_basics():
    b = quoridor_rs.Board()
    moves = b.legal_moves()
    assert len(moves) == 131
    assert b.decode(0) == ("pawn", 1, 4)
    enc = b.encode()
    assert enc.shape == (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert enc[0, 0, 4] == 1 and enc[1, 8, 4] == 1
    with pytest.raises(ValueError):
        b.push(5)  # SS: off the board
    b.push(0)
    assert b.turn == 1 and b.ply == 1
    # Player 1's frame flips rows, so its own pawn (row 8) shows at row 0.
    assert b.encode()[0, 0, 4] == 1


def random_board(rng, plies):
    b = quoridor_rs.Board()
    for _ in range(plies):
        moves = b.legal_moves()
        if not moves:
            break
        b.push(moves[rng.integers(len(moves))])
    return b


def test_replay_mirror_matches_mirrored_board():
    rng = np.random.default_rng(0)
    perms = (quoridor_rs.mirror_state_perm(), quoridor_rs.mirror_policy_perm())
    buf = ReplayBuffer(16, mirror=perms)
    for plies in range(16):
        b = random_board(rng, plies * 3)
        pi = np.zeros(POLICY_SIZE, dtype=np.float32)
        pi[b.legal_moves()] = 1.0
        buf.add_many([(b.encode(), pi, 0.5)])
        m = b.mirrored()
        mpi = np.zeros(POLICY_SIZE, dtype=np.float32)
        mpi[m.legal_moves()] = 1.0
        state_perm, policy_perm = map(np.asarray, perms)
        assert np.array_equal(b.encode().reshape(-1)[state_perm], m.encode().reshape(-1))
        assert np.array_equal(pi[policy_perm], mpi)
    states, pis, zs = buf.sample(64, rng)
    assert states.shape == (64, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert pis.shape == (64, POLICY_SIZE) and np.all(zs == 0.5)
