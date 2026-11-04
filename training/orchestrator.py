"""Self-play training orchestrator (port of chess-ai-engine's orchestrator.py at
commit 53b1b83).

Runs the trainer in this (main) process and spawns ONE self-play process
running the native quoridor_rs pipeline with the net on the GPU (selfplay.py).
That process pushes finished games onto a bounded queue; the trainer drains it
into a ring ReplayBuffer, runs SGD paced to the data rate, and republishes
weights to serving_weights.pt, which the self-play side reloads in place.

Every --snapshot-every it writes snapshots/<stem>.pt + .onnx and launches
evaluate.py detached on the new snapshot (skipped if the previous eval is still
running). latest.pt (weights + optimizer + clock + RNG) allows resuming.

IMPORTANT: module-level imports are kept torch-free. `spawn` re-imports this
module in the child, so importing torch here would load it twice for nothing.
All torch imports live inside main().
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import queue as pyqueue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from selfplay import run_selfplay_pipeline


def parse_duration(spec):
    """Parse a duration string like '12h', '30m', '300s' into seconds."""
    s = str(spec).strip()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start a fresh run from this checkpoint (weights "
                         "[+optimizer], fresh clock). Only on cold start.")
    ap.add_argument("--snapshot-every", default="30m")
    ap.add_argument("--save-latest-every", default="300s")
    ap.add_argument("--publish-every", default="15s",
                    help="How often the trainer republishes weights for the "
                         "self-play process to reload.")
    # self-play
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--queue-max", type=int, default=256)
    ap.add_argument("--bucket-size", type=int, default=512,
                    help="GPU batch width per self-play forward (in-flight games "
                         "self-regulate to ~2x this).")
    ap.add_argument("--n-threads", type=int, default=16,
                    help="Self-play MCTS worker threads.")
    # training
    ap.add_argument("--buffer-capacity", type=int, default=200_000)
    ap.add_argument("--min-buffer-for-train", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--target-reuse", type=float, default=4.0,
                    help="Avg times each generated sample is used in training. "
                         "Paces SGD to the self-play data rate so the trainer "
                         "doesn't starve self-play of GPU, and bounds overfitting.")
    ap.add_argument("--max-steps-per-cycle", type=int, default=64,
                    help="Cap on SGD steps per loop iteration (smooths bursts).")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=float, default=30.0)
    ap.add_argument("--eval-args", default="",
                    help="Extra arguments passed through to evaluate.py.")
    return ap.parse_args()


def main():
    args = parse_args()

    # torch-heavy imports live here so the spawn-reimported child stays light.
    import torch

    import quoridor_rs
    from export import export_module
    from net import N_BLOCKS, N_FILTERS, QuoridorNet, n_params
    from train import ReplayBuffer, train_step

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_interval = parse_duration(args.snapshot_every)
    save_latest_interval = parse_duration(args.save_latest_every)
    publish_interval = parse_duration(args.publish_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = QuoridorNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_capacity,
                       mirror=(quoridor_rs.mirror_state_perm(),
                               quoridor_rs.mirror_policy_perm()))
    rng = np.random.default_rng(args.seed)
    config = {"n_blocks": N_BLOCKS, "n_filters": N_FILTERS}

    # ----- Resume from latest.pt, else warm-start from --init-from, else cold -----
    base_elapsed = 0.0
    total_games = 0
    total_train_steps = 0
    next_snapshot_at = 0.0
    loaded_interval = None
    latest_path = out_dir / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        opt.load_state_dict(ckpt["opt"])
        base_elapsed = ckpt.get("elapsed_sec", 0.0)
        total_games = ckpt.get("games", 0)
        total_train_steps = ckpt.get("train_steps", 0)
        next_snapshot_at = ckpt.get("next_snapshot_at", 0.0)
        loaded_interval = ckpt.get("snapshot_interval")
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "np_rng" in ckpt:
            rng.bit_generator.state = ckpt["np_rng"]
        print(f"resumed from {latest_path} (elapsed={base_elapsed/3600:.2f}h "
              f"games={total_games} steps={total_train_steps})", flush=True)
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["weights"])
        if "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError) as e:
                print(f"warn: skipped optimizer state from --init-from ({e})",
                      flush=True)
        torch.manual_seed(args.seed)
        print(f"warm start from {args.init_from} (fresh clock; seed={args.seed})",
              flush=True)
    else:
        torch.manual_seed(args.seed)
        print(f"cold start (seed={args.seed})", flush=True)

    print(f"net: blocks={N_BLOCKS} filters={N_FILTERS} params={n_params(net):,}",
          flush=True)

    # ----- Checkpoint / snapshot / eval helpers -----
    def _atomic_save(path, obj):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    serving_path = out_dir / "serving_weights.pt"

    def publish_weights():
        _atomic_save(serving_path, {"weights": net.state_dict(), "config": config})

    def save_latest():
        _atomic_save(latest_path, {
            "weights": net.state_dict(),
            "opt": opt.state_dict(),
            "elapsed_sec": elapsed(),
            "games": total_games,
            "train_steps": total_train_steps,
            "next_snapshot_at": next_snapshot_at,
            "snapshot_interval": snapshot_interval,
            "torch_rng": torch.get_rng_state(),
            "np_rng": rng.bit_generator.state,
            "config": config,
        })

    def save_snapshot():
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        minutes = int(elapsed() / 60)
        utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")
        stem = f"snap_m{minutes:06d}_{utc}"
        pt_path = snap_dir / f"{stem}.pt"
        _atomic_save(pt_path, {
            "weights": net.state_dict(),
            "config": config,
            "stats": {"elapsed_sec": elapsed(), "games": total_games,
                      "train_steps": total_train_steps, "buffer_size": len(buf)},
        })
        export_module(net, str(snap_dir / f"{stem}.onnx"))
        print(f"[snapshot] {stem}", flush=True)
        return pt_path

    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # A finished-but-unreaped child lingers as a zombie whose PID-table
        # entry keeps os.kill(pid, 0) succeeding. Treat zombies as dead.
        try:
            with open(f"/proc/{pid}/stat") as f:
                data = f.read()
            if data[data.rindex(")") + 1:].split()[0] == "Z":
                return False
        except (OSError, ValueError, IndexError):
            pass
        return True

    def launch_eval(pt_path):
        prev = launch_eval.proc
        if prev is not None and prev.poll() is not None:
            launch_eval.proc = None
        lock = out_dir / "eval.lock"
        if lock.exists():
            try:
                prev_pid = int(lock.read_text().split()[0])
                if _pid_alive(prev_pid):
                    print(f"[eval] prior eval (pid {prev_pid}) still running; "
                          f"skipping {pt_path.name}", flush=True)
                    return
            except (ValueError, IndexError, OSError):
                pass
        training_dir = Path(__file__).resolve().parent
        logf = open(out_dir / "eval.log", "a")
        try:
            proc = subprocess.Popen(
                [sys.executable, "evaluate.py",
                 "--run-dir", str(out_dir), "--candidate", str(pt_path),
                 *args.eval_args.split()],
                cwd=str(training_dir), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            logf.close()
        launch_eval.proc = proc
        lock.write_text(f"{proc.pid} {time.time()}")
        print(f"[eval] launched evaluate.py pid={proc.pid} for {pt_path.name}",
              flush=True)

    launch_eval.proc = None

    # ----- Spawn the self-play process -----
    publish_weights()

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=args.queue_max)
    stop = ctx.Event()

    def start_selfplay():
        p = ctx.Process(
            target=run_selfplay_pipeline,
            args=(out_queue, stop, args.seed + 1000 + total_games, args.sims,
                  str(serving_path), args.n_threads, args.bucket_size, str(device)),
        )
        p.start()
        return p

    selfplay_proc = start_selfplay()

    # ----- Clock / snapshot schedule -----
    start = time.time()
    last_log = start
    last_latest_save = start
    last_publish = start

    def elapsed():
        return base_elapsed + (time.time() - start)

    if loaded_interval is not None and abs(loaded_interval - snapshot_interval) > 1e-6:
        next_snapshot_at = (int(elapsed() // snapshot_interval) + 1) * snapshot_interval
        print(f"snapshot interval changed; realigned next snapshot to "
              f"elapsed={next_snapshot_at/3600:.2f}h", flush=True)
    while next_snapshot_at <= elapsed():
        next_snapshot_at += snapshot_interval
    print(f"snapshot every {args.snapshot_every} "
          f"(next at elapsed={next_snapshot_at/3600:.2f}h)", flush=True)

    def drain(item):
        nonlocal total_games, gen_samples, last_avg_plies
        rows, n_games, avg_plies = item
        buf.add_many(rows)
        total_games += n_games
        gen_samples += len(rows)
        last_avg_plies = avg_plies

    last_avg_plies = 0.0
    last_loss = None
    gen_samples = 0       # rows generated this session (for reuse pacing)
    session_steps = 0     # SGD steps taken this session (for reuse pacing)
    net.train()
    try:
        while True:
            if not selfplay_proc.is_alive():
                print("warn: self-play process died; restarting", flush=True)
                selfplay_proc = start_selfplay()

            if elapsed() >= next_snapshot_at:
                pt_path = save_snapshot()
                save_latest()
                last_latest_save = time.time()
                next_snapshot_at += snapshot_interval
                launch_eval(pt_path)
                continue

            got_any = False
            for _ in range(64):
                try:
                    drain(out_queue.get_nowait())
                except pyqueue.Empty:
                    break
                got_any = True
            if len(buf) < args.min_buffer_for_train and not got_any:
                try:
                    drain(out_queue.get(timeout=1.0))
                except pyqueue.Empty:
                    pass

            # SGD, paced to the data rate: keep cumulative samples-trained near
            # target_reuse x samples-generated.
            cycle_tr_time = 0.0
            allowed = int(gen_samples * args.target_reuse / args.batch_size) - session_steps
            n_steps = max(0, min(allowed, args.max_steps_per_cycle))
            if len(buf) >= args.min_buffer_for_train and n_steps > 0:
                train_start = time.time()
                losses = []
                for _ in range(n_steps):
                    losses.append(train_step(net, opt, buf.sample(args.batch_size, rng), device))
                    total_train_steps += 1
                    session_steps += 1
                cycle_tr_time = time.time() - train_start
                last_loss = {
                    "loss": round(float(np.mean([l["loss"] for l in losses])), 4),
                    "policy": round(float(np.mean([l["policy_loss"] for l in losses])), 4),
                    "value": round(float(np.mean([l["value_loss"] for l in losses])), 4),
                }
            elif not got_any:
                time.sleep(0.05)

            now = time.time()
            if now - last_publish >= publish_interval:
                publish_weights()
                last_publish = now
            if now - last_latest_save >= save_latest_interval:
                save_latest()
                last_latest_save = now
            if now - last_log >= args.log_every:
                print(json.dumps({
                    "t": round(elapsed(), 1),
                    "games": total_games,
                    "buf": len(buf),
                    "tr_steps": total_train_steps,
                    "reuse": round(session_steps * args.batch_size / max(1, gen_samples), 2),
                    "qsize": out_queue.qsize(),
                    "tr_sec": round(cycle_tr_time, 2),
                    "avg_plies": round(last_avg_plies, 1),
                    "loss": last_loss,
                    "next_snap_in": round(next_snapshot_at - elapsed(), 1),
                }), flush=True)
                last_log = now
    finally:
        print("shutting down: signaling self-play", flush=True)
        stop.set()
        selfplay_proc.join(timeout=10)
        if selfplay_proc.is_alive():
            selfplay_proc.terminate()


if __name__ == "__main__":
    main()
