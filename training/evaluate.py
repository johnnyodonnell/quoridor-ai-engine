"""Self-play evaluation, Elo rating, and model serving (port of
chess-ai-engine's evaluate.py).

Each invocation evaluates one candidate snapshot against the active pool
(top performers + a fixed random anchor + frozen ex-snapshot anchors spread
across the strength range), refits a lightweight Elo over all accumulated
match results (random pinned at 0), updates the pool, and promotes the
highest-rated model to `<run-dir>/best.onnx` if it clears the current served
model by a confidence margin.

Match play runs in quoridor_rs.MatchEngine (batched MCTS across all games of a
pairing); this process only runs the two nets' forwards.

    python evaluate.py --run-dir runs/run1 --candidate runs/run1/snapshots/<snap>.pt

orchestrator.py launches it detached after each snapshot.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import quoridor_rs
from export import export
from net import POLICY_SIZE, load_net

RANDOM = "random"  # reserved name for the fixed floor anchor (Elo pinned at 0)


# ---------------------------------------------------------------------------
# Match play
# ---------------------------------------------------------------------------
@torch.no_grad()
def _forward(net, enc, device):
    x = torch.from_numpy(np.ascontiguousarray(enc)).to(device)
    logits, values = net(x)
    return logits.float().cpu().numpy(), values.float().cpu().numpy()


def play_match(net_a, net_b, n_games, sims, opening_plies, seed, device):
    """Play `n_games` between two players (a net, or None for random),
    alternating who moves first. Returns (wins, draws, losses) for A."""
    eng = quoridor_rs.MatchEngine(n_games, sims, opening_plies, seed,
                                  net_a is None, net_b is None)
    nets = (net_a, net_b)
    while True:
        enc, who = eng.leaves()
        m = len(who)
        if m == 0:
            break
        logits = np.empty((m, POLICY_SIZE), dtype=np.float32)
        values = np.empty(m, dtype=np.float32)
        for k in (0, 1):
            rows = who == k
            if rows.any():
                logits[rows], values[rows] = _forward(nets[k], enc[rows], device)
        eng.apply(logits, values)

    res = eng.results()
    wins = sum(r == 1 for r in res)
    losses = sum(r == -1 for r in res)
    return wins, n_games - wins - losses, losses


# ---------------------------------------------------------------------------
# Elo fit (Bradley-Terry, coordinate-Newton; random pinned, L2-regularized)
# ---------------------------------------------------------------------------
def fit_elo(names, games, fixed=None, reg=1e-4, iters=400):
    """`games`: list of (a, b, score_a, n) — in n games between a and b, a
    scored score_a points (win=1, draw=0.5). Returns {name: rating}.

    random (or any name in `fixed`) is held at its fixed rating to pin the
    scale; reg pulls ratings toward 0 so a 100% sweep stays finite.
    """
    fixed = fixed or {RANDOM: 0.0}
    R = {n: fixed.get(n, 0.0) for n in names}
    q = math.log(10) / 400.0

    adj = {n: [] for n in names}
    for a, b, score_a, n in games:
        if n <= 0:
            continue
        adj[a].append((b, score_a, n))
        adj[b].append((a, n - score_a, n))

    for _ in range(iters):
        for p in names:
            if p in fixed:
                continue
            g = h = 0.0
            for opp, score_p, n in adj[p]:
                e = 1.0 / (1.0 + 10 ** ((R[opp] - R[p]) / 400.0))
                g += q * (score_p - n * e)
                h += q * q * n * e * (1.0 - e)
            g -= reg * R[p]
            h += reg
            if h > 1e-12:
                R[p] += g / h
    for n, v in fixed.items():
        R[n] = v
    return R


# ---------------------------------------------------------------------------
# Pool / serving
# ---------------------------------------------------------------------------
def _load_pool(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"served": None, "served_rating": None, "models": {}, "results": []}


def _save_pool(path, pool):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(pool, indent=2))
    tmp.replace(path)


def _select_anchors(ratings, models, n_top, n_anchors):
    """Active opponents = top `n_top` rated + random + `n_anchors` frozen
    snapshots whose ratings most evenly cover (0, rating of the n_top-th)."""
    ranked = sorted(
        (m for m in models if m != RANDOM),
        key=lambda m: ratings.get(m, 0.0),
        reverse=True,
    )
    top = ranked[:n_top]
    active = list(top) + [RANDOM]

    below = ranked[n_top:]
    if below:
        ceiling = ratings.get(top[-1], 0.0) if top else max(
            (ratings.get(m, 0.0) for m in below), default=0.0
        )
        # Even target rungs across (0, ceiling); pick the closest unused model.
        k = min(n_anchors, len(below))
        targets = [ceiling * (i + 1) / (k + 1) for i in range(k)]
        chosen = []
        pool = list(below)
        for t in targets:
            best = min(pool, key=lambda m: abs(ratings.get(m, 0.0) - t))
            chosen.append(best)
            pool.remove(best)
        active += chosen
    return active, top


def _rel(path, run_dir):
    return str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path)


def evaluate(run_dir, candidate, games_per_pair, sims, opening_plies,
             n_top, n_anchors, serve_margin, seed):
    t0 = time.time()
    run_dir = Path(run_dir).resolve()
    pool_path = run_dir / "pool.json"
    pool = _load_pool(pool_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cand_pt = Path(candidate).resolve()
    cand_name = cand_pt.stem
    print(f"[eval] candidate={cand_name} device={device}", flush=True)

    pool["models"][cand_name] = {
        "pt": _rel(cand_pt, run_dir),
        "onnx": _rel(cand_pt.with_suffix(".onnx"), run_dir),
        "rating": None,
    }
    pool["models"].setdefault(RANDOM, {"pt": None, "onnx": None, "rating": 0.0})

    prior = {m: (pool["models"][m].get("rating") or 0.0) for m in pool["models"]}
    active, _ = _select_anchors(prior, list(pool["models"]), n_top, n_anchors)
    opponents = [m for m in active if m != cand_name] or [RANDOM]
    print(f"[eval] opponents={opponents}", flush=True)

    cand_net = load_net(cand_pt, device)
    cand_vs_random = None
    for i, opp in enumerate(opponents):
        opp_net = None if opp == RANDOM else load_net(run_dir / pool["models"][opp]["pt"], device)
        t = time.time()
        w, d, l = play_match(cand_net, opp_net, games_per_pair, sims,
                             opening_plies, seed + i, device)
        score_a = w + 0.5 * d
        pool["results"].append({"a": cand_name, "b": opp, "score_a": score_a, "n": w + d + l})
        if opp == RANDOM:
            cand_vs_random = score_a / max(1, w + d + l)
        print(f"[eval] {cand_name} vs {opp}: W{w} D{d} L{l} ({time.time() - t:.0f}s)",
              flush=True)

    names = list(pool["models"])
    games = [(r["a"], r["b"], r["score_a"], r["n"]) for r in pool["results"]]
    ratings = fit_elo(names, games)
    for m in names:
        pool["models"][m]["rating"] = round(ratings.get(m, 0.0), 1)

    _, top = _select_anchors(ratings, names, n_top, n_anchors)
    leader = top[0] if top else None
    pool["top"] = top
    print("[eval] ratings: " + ", ".join(
        f"{m}={pool['models'][m]['rating']}" for m in
        sorted(names, key=lambda m: ratings.get(m, 0.0), reverse=True)
    ), flush=True)

    served = pool.get("served")
    # Compare against the served model's CURRENT refit rating: every refit
    # drifts the scale, so a frozen threshold could become unbeatable.
    served_now = ratings.get(served) if served else None
    sane = (cand_vs_random is None) or (cand_vs_random >= 0.6)
    if leader and sane:
        lead_r = ratings[leader]
        if served is None or lead_r > (served_now or -1e9) + serve_margin:
            _serve(run_dir, pool, leader)
            pool["served"], pool["served_rating"] = leader, round(lead_r, 1)
            print(f"[eval] SERVED {leader} (rating={lead_r:.1f})", flush=True)
        else:
            print(f"[eval] kept {served} (rating={served_now:.1f}); "
                  f"leader {leader}={lead_r:.1f} within margin", flush=True)
    elif not sane:
        print(f"[eval] candidate failed random sanity gate "
              f"(score vs random={cand_vs_random:.2f}); not serving", flush=True)

    _save_pool(pool_path, pool)
    print(f"[eval] done in {time.time() - t0:.0f}s", flush=True)


def _serve(run_dir, pool, name):
    """Copy the model's onnx (+pt) to best.onnx/best.pt, exporting if needed."""
    info = pool["models"][name]
    pt = run_dir / info["pt"]
    onnx = run_dir / info["onnx"]
    if not onnx.exists():
        export(str(pt), str(onnx))
    shutil.copyfile(onnx, run_dir / "best.onnx")
    shutil.copyfile(pt, run_dir / "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--candidate", required=True, help="path to the snapshot .pt")
    ap.add_argument("--games-per-pair", type=int, default=100)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=10)
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--n-anchors", type=int, default=3)
    ap.add_argument("--serve-margin", type=float, default=35.0,
                    help="Elo the leader must beat the served model by to be "
                         "promoted (≈ rating standard error at 100 games).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    evaluate(
        args.run_dir, args.candidate, args.games_per_pair, args.sims,
        args.opening_plies, args.n_top, args.n_anchors, args.serve_margin,
        args.seed,
    )


if __name__ == "__main__":
    main()
