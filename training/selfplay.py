"""Self-play process: runs the native decoupled pipeline (quoridor_rs
PipelineEngine) with the net on the GPU in this process.

`n_threads` Rust workers push MCTS leaves into two inference buckets; the single
consumer calls `forward` (below) on each full bucket. Finished games come back
through `submit_game` and go onto the trainer's bounded queue as
(rows, n_games, avg_plies).

The net reloads its weights from `weights_path` whenever the trainer
republishes it, IN PLACE so the compiled graph is reused.
"""

import os
import queue as _queue
import time

import numpy as np


def run_selfplay_pipeline(out_queue, stop_event, seed, sims, weights_path,
                          n_threads, bucket_size, device_str="cuda", reload_every=5.0):
    import torch

    import quoridor_rs
    from net import load_net

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    # QUORIDOR_COMPILE=1 wraps the net in torch.compile: the net is tiny, so eager
    # mode is dominated by per-op dispatch; compile fuses it into a few kernels.
    # QUORIDOR_BF16=1 runs the forward in bf16 (MCTS priors don't need fp32).
    compile_net = os.environ.get("QUORIDOR_COMPILE", "1") == "1"
    dtype = torch.bfloat16 if os.environ.get("QUORIDOR_BF16", "1") == "1" else torch.float32

    eager = load_net(weights_path, device).to(dtype)
    state = {"net": torch.compile(eager) if compile_net else eager,
             "mtime": os.path.getmtime(weights_path),
             "last_check": time.time(),
             "reloads": 0}
    print(f"selfplay: compile={compile_net} dtype={dtype} bucket={bucket_size} "
          f"threads={n_threads} sims={sims}", flush=True)

    def maybe_reload():
        now = time.time()
        if now - state["last_check"] < reload_every:
            return
        state["last_check"] = now
        try:
            m = os.path.getmtime(weights_path)
            if m > state["mtime"]:
                ckpt = torch.load(weights_path, map_location=device, weights_only=False)
                # load_state_dict copies (and casts) into the existing params, so
                # the compiled graph keeps working without a retrace.
                eager.load_state_dict(ckpt["weights"])
                state["mtime"] = m
                state["reloads"] += 1
                if state["reloads"] % 40 == 1:
                    print(f"selfplay: reloaded weights ({state['reloads']} so far)", flush=True)
        except (OSError, RuntimeError, KeyError, EOFError):
            pass  # mid-write / transient — retry next check

    @torch.no_grad()
    def forward(batch):
        maybe_reload()
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(device, dtype)
        logits, values = state["net"](x)
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    # Orphans get re-parented to a subreaper (systemd --user), not necessarily
    # PID 1, so detect the trainer's death as any change of parent.
    parent_pid = os.getppid()

    def should_stop():
        return stop_event.is_set() or os.getppid() != parent_pid

    def submit_game(rows):
        item = (rows, 1, len(rows))
        while not should_stop():
            try:
                out_queue.put(item, timeout=0.5)
                return
            except _queue.Full:
                pass

    engine = quoridor_rs.PipelineEngine(sims, seed, True, n_threads, bucket_size)
    engine.run(forward, submit_game, should_stop)
