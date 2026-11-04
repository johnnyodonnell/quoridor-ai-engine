"""Dump random games played by quoridor_rs (moves, legal moves and encodings
at every ply) so scripts/check_js_encoding.mjs can check the browser engine's
Position against the Rust board.

    python dump_encoding_fixture.py [n_games=60] [out=fixtures/encoding.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

import quoridor_rs


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "fixtures/encoding.json")
    rng = np.random.default_rng(7)
    games = []
    for _ in range(n_games):
        b = quoridor_rs.Board()
        plies = []
        while b.outcome() is None:
            moves = b.legal_moves()
            # Lean on pawn moves so games reach jumps and finish; walls ~30%.
            pawn = [m for m in moves if m < 12]
            walls = [m for m in moves if m >= 12]
            pool = walls if walls and rng.random() < 0.3 else pawn
            move = int(pool[rng.integers(len(pool))])
            plies.append({
                "legal": sorted(moves),
                "enc": [round(float(x), 6) for x in b.encode().reshape(-1)],
                "move": move,
                "decoded": list(b.decode(move)),
            })
            b.push(move)
        games.append({"plies": plies, "outcome": b.outcome()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(games))
    print(f"wrote {out}: {len(games)} games, {sum(len(g['plies']) for g in games)} plies")


if __name__ == "__main__":
    main()
