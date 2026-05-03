"""
train_phase2.py — Phase 2: Self-play training with parallel game collection.

All 4 seats use the same shared policy.  The network learns to compete
against a competent opponent rather than exploiting random mistakes.

Parallel collection:
    N_WORKERS subprocesses each run N_STEPS // N_WORKERS game steps
    simultaneously, then return numpy buffers.  The main process
    concatenates them and runs the PPO update.  This gives ~N_WORKERS×
    speedup on the game-engine bottleneck (~3× net on a 4-core machine).

    Adjust N_WORKERS to match your physical CPU core count.

Input  : phase1_policy.pt  (written by train_phase1.py)
           — if not found, starts from random weights with a warning.
         phase2_policy.pt  (optional — resumes mid-phase if it exists)
Output : phase2_policy.pt  (pass this to train_phase3.py to continue)

Run from the Game/ directory:
    python train_phase2.py
"""

import sys, os, io
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(__file__))

import torch
from collections import defaultdict

from catan_env import CatanEnv
from ppo_utils import (
    NUM_PLAYERS, N_STEPS, LOG_EVERY,
    CKPT_PHASE1, CKPT_PHASE2,
    REWARD_CONFIG_PHASE2 as REWARD_CONFIG,
    make_policy, compute_gae, ppo_update,
    _act_type,
)
import ppo_utils as _ppo_utils
_ppo_utils.ENT_COEF = 0.05   # middle ground: enough exploration for P1/P2/P3 without swamping their gradients

# ── Phase 2 specific settings ────────────────────────────────────────────────
NUM_UPDATES = 200  # how many PPO updates to run in this phase
N_WORKERS   = 4     # parallel collection workers — tune to physical CPU cores


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_dir() -> Path:
    return Path(__file__).resolve().parent


def _checkpoint_base(path_str: str) -> tuple[str, str]:
    ckpt_path = Path(path_str)
    return ckpt_path.stem, ckpt_path.suffix or ".pt"


def _latest_checkpoint_path(path_str: str) -> "Path | None":
    base_stem, suffix = _checkpoint_base(path_str)
    ckpt_dir = _checkpoint_dir()
    candidates = sorted(ckpt_dir.glob(f"{base_stem}*{suffix}"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _next_checkpoint_path(path_str: str) -> Path:
    base_stem, suffix = _checkpoint_base(path_str)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _checkpoint_dir() / f"{base_stem}_{timestamp}{suffix}"


# ── Worker function (must be at module level for Windows pickling) ────────────

def _worker_collect(args: tuple) -> dict:
    """
    Run n_steps of self-play in a subprocess and return a numpy buffer.

    Workers are stateless: they receive the current policy weights as bytes,
    create a fresh env + policy, collect steps, and return raw numpy arrays.
    The main process concatenates results from all workers before the update.
    """
    weights_bytes, n_steps, num_players, reward_config = args

    import sys, os, io
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    import numpy as np
    from collections import defaultdict
    from catan_env import CatanEnv
    from ppo_utils import make_policy

    # Workers process one observation at a time — intra-op parallelism only
    # causes thread contention when multiple workers run on the same machine.
    torch.set_num_threads(1)

    env = CatanEnv(num_players=num_players, reward_shaping=True, **reward_config)
    policy, _ = make_policy(env)
    policy.load_state_dict(torch.load(io.BytesIO(weights_bytes), weights_only=True))
    policy.eval()

    obs_list:   list = []
    masks_list: list = []
    actions:    list = []
    log_probs:  list = []
    values:     list = []
    rewards:    list = []
    dones:      list = []
    wins: dict = defaultdict(int)

    obs_np, mask_np = env.reset()

    for _ in range(n_steps):
        pid    = env.current_player
        obs_t  = torch.from_numpy(obs_np).float()
        mask_t = torch.from_numpy(mask_np)

        with torch.no_grad():
            logits, value = policy(obs_t)
            logits[~mask_t] = -1e9
            probs  = torch.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

        rew, done = env.step(action.item())

        next_pid = env.current_player if not done else -1

        obs_list.append(obs_np)
        masks_list.append(mask_np)
        actions.append(action.item())
        log_probs.append(dist.log_prob(action).item())
        values.append(value.squeeze().item())
        rewards.append(float(rew[pid]))
        # Mark a GAE boundary whenever the acting player changes — prevents
        # bootstrapping P0's value off P1's next-state value, which corrupts
        # advantages for all non-first-mover seats.
        player_switch = (next_pid != pid) and not done
        dones.append(1.0 if done or player_switch else 0.0)

        if done:
            if env.winner is not None:
                wins[env.winner] += 1

        obs_np, mask_np = env.reset() if done else env.observe()

    return {
        "obs":       np.array(obs_list,   dtype=np.float32),
        "masks":     np.array(masks_list, dtype=bool),
        "actions":   np.array(actions,    dtype=np.int64),
        "log_probs": np.array(log_probs,  dtype=np.float32),
        "values":    np.array(values,     dtype=np.float32),
        "rewards":   rewards,
        "dones":     dones,
        "wins":      dict(wins),
    }


# ── Main: setup + training loop ───────────────────────────────────────────────
# Must be guarded for Windows multiprocessing (spawn method).

if __name__ == "__main__":
    import numpy as np

    env           = CatanEnv(num_players=NUM_PLAYERS, reward_shaping=True, **REWARD_CONFIG)
    policy, optim = make_policy(env)

    resume_path = _latest_checkpoint_path(CKPT_PHASE2)
    save_path   = _next_checkpoint_path(CKPT_PHASE2)

    # Priority: resume mid-phase > start from phase 1 > start from scratch
    if resume_path is not None:
        try:
            policy.load_state_dict(torch.load(resume_path, weights_only=True))
            print(f"Resumed from {resume_path.name}")
        except RuntimeError as e:
            print(f"WARNING: Could not load checkpoint {resume_path.name}: {e}")
            print("Starting Phase 2 from scratch (checkpoint architecture mismatch).")
    else:
        phase1_path = _latest_checkpoint_path(CKPT_PHASE1)
        if phase1_path is not None:
            try:
                policy.load_state_dict(torch.load(phase1_path, weights_only=True))
                print(f"Loaded Phase 1 weights from {phase1_path.name}")
            except RuntimeError as e:
                print(f"WARNING: Could not load Phase 1 checkpoint {phase1_path.name}: {e}")
                print("Starting Phase 2 from random weights (checkpoint architecture mismatch).")
        else:
            print(f"WARNING: no {Path(CKPT_PHASE1).stem} checkpoint found. "
                  "Run train_phase1.py first for best results.")
            print("Starting Phase 2 from random weights.\n")

    n_per_worker = N_STEPS // N_WORKERS
    print(f"Phase 2: full self-play for {NUM_UPDATES} updates "
          f"({N_WORKERS} workers × {n_per_worker} steps = {N_STEPS} steps/update).\n")

    win_counts: dict = defaultdict(int)
    act_totals: dict = defaultdict(int)

    with mp.Pool(N_WORKERS) as pool:
        try:
            for update in range(1, NUM_UPDATES + 1):
                # Serialize current policy weights once per update
                buf_io = io.BytesIO()
                torch.save(policy.state_dict(), buf_io)
                weights_bytes = buf_io.getvalue()

                worker_args = [
                    (weights_bytes, n_per_worker, NUM_PLAYERS, REWARD_CONFIG)
                ] * N_WORKERS

                results = pool.map(_worker_collect, worker_args)

                # Concatenate worker buffers into one flat buffer
                all_obs       = np.concatenate([r["obs"]       for r in results], axis=0)
                all_masks     = np.concatenate([r["masks"]     for r in results], axis=0)
                all_actions   = np.concatenate([r["actions"]   for r in results])
                all_log_probs = np.concatenate([r["log_probs"] for r in results])
                all_values    = np.concatenate([r["values"]    for r in results])
                all_rewards   = sum([r["rewards"] for r in results], [])
                all_dones     = sum([r["dones"]   for r in results], [])

                for r in results:
                    for pid, count in r["wins"].items():
                        win_counts[pid] += count

                # Count action types from this update's collected actions
                for a in all_actions:
                    act_totals[_act_type(int(a))] += 1

                # Convert to tensor format expected by compute_gae / ppo_update
                val_tensors = list(torch.from_numpy(all_values).unbind())
                buf = {
                    "obs":       [torch.from_numpy(all_obs)],
                    "masks":     [torch.from_numpy(all_masks)],
                    "actions":   list(torch.from_numpy(all_actions).unbind()),
                    "log_probs": list(torch.from_numpy(all_log_probs).unbind()),
                    "values":    val_tensors,
                    "rewards":   all_rewards,
                    "dones":     all_dones,
                }

                adv, ret   = compute_gae(buf["rewards"], buf["values"], buf["dones"])
                total_loss = ppo_update(buf, adv, ret, policy, optim)

                if update % LOG_EVERY == 0:
                    games    = sum(win_counts.values())
                    total    = games or 1
                    rates    = {p: win_counts[p] / total for p in range(NUM_PLAYERS)}
                    rate_str = " | ".join(
                        f"p{p} {rates.get(p, 0):.2f}"
                        for p in range(NUM_PLAYERS)
                    )
                    n_acts = sum(act_totals.values()) or 1
                    acts   = "  ".join(f"{k}:{act_totals[k]/n_acts:.2f}" for k in
                                       ["roll","end","settle","road","city","trade","robber","buydev"])
                    print(f"Update {update:4d}/{NUM_UPDATES} | loss {total_loss:9.4f} | entropy {torch.mean(-torch.stack(buf['log_probs'])).item():.4f} |"
                          f"games {games:3d} | {rate_str}")
                    print(f"  actions: {acts}")
                    win_counts.clear()
                    act_totals.clear()

        except KeyboardInterrupt:
            print("\nInterrupted — terminating workers and saving checkpoint...")
            pool.terminate()
            pool.join()

    torch.save(policy.state_dict(), save_path)
    print(f"\nPhase 2 complete. Saved {save_path.name}")
    print(f"Next step: run  python train_phase3.py")
