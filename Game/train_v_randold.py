"""
train_phase3.py — Phase 3: League play training with parallel game collection.

Every SNAPSHOT_EVERY updates, a frozen copy (snapshot) of the current
policy is added to the league pool.  Opponent seats randomly draw from
the pool (old versions) or use the live policy, so the agent must handle
both old and new playstyles simultaneously.

This prevents strategy collapse — without a pool, the policy can overfit
to beating its current self while forgetting how to beat earlier strategies.

Parallel collection:
    N_WORKERS subprocesses each run N_STEPS // N_WORKERS game steps
    simultaneously.  Each worker receives the live policy weights AND the
    current league pool snapshot so opponent seats can use frozen policies.
    Adjust N_WORKERS to match your physical CPU core count.

Input  : phase2_policy.pt  (written by train_phase2.py)
           — if not found, starts from phase1_policy.pt, then random weights.
         phase3_policy.pt  (optional — resumes mid-phase if it exists)
         league_pool.pt    (optional — resumes the snapshot pool if it exists)
Output : phase3_policy.pt
         league_pool.pt

Run from the Game/ directory:
    python train_phase3.py
"""

import sys, os, copy, io
import random as pyrandom
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(__file__))

import torch
from collections import defaultdict

from catan_env import CatanEnv
from ppo_utils import (
    NUM_PLAYERS, HIDDEN_SIZE, N_STEPS, LOG_EVERY,
    CKPT_PHASE1, CKPT_PHASE2, CKPT_PHASE3, CKPT_LEAGUE,
    make_policy, compute_gae, ppo_update,
)

# ── Phase 3 specific settings ─────────────────────────────────────────────────
NUM_UPDATES    = 2000   # how many PPO updates to run in this phase
SNAPSHOT_EVERY = 100    # freeze a snapshot into the pool every N updates
LEAGUE_PROB    = 0.5    # chance any opponent seat draws from the pool
N_WORKERS      = 4      # parallel collection workers — tune to physical CPU cores
REWARD_CONFIG = {
    "public_vp_reward": 0.3,
    "road_reward": 0.05,
    "buy_dev_reward": 0.10,
    "win_reward": 5.0,
    "loss_penalty": 3.0,  # moderate — league opponents are older versions of itself
    "setup_settle_reward": 0.5,
    "robber_block_reward": 0.1,
    "monopoly_reward": 0.3,
    "yop_build_reward": 0.15,
}


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
    Run n_steps of league play in a subprocess and return a numpy buffer.

    Player 0 always uses the live policy (its steps are stored for training).
    Opponent seats (1-3) flip a coin: LEAGUE_PROB → random frozen snapshot,
    else live policy.  Snapshot steps are NOT stored in the gradient buffer.

    Workers pre-load all league snapshots once on entry (faster than
    re-creating a CatanPolicy per snapshot per step as league_action does).
    """
    (weights_bytes, pool_bytes, n_steps,
     num_players, reward_config, league_prob, hidden_size) = args

    import sys, os, io
    import random as pyrandom
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    import numpy as np
    from catan_env import CatanEnv
    from ppo_utils import make_policy
    from policy import CatanPolicy

    torch.set_num_threads(1)

    env = CatanEnv(num_players=num_players, reward_shaping=True, **reward_config)

    # Live policy
    policy, _ = make_policy(env)
    policy.load_state_dict(torch.load(io.BytesIO(weights_bytes), weights_only=True))
    policy.eval()

    # League pool — pre-load each snapshot into a ready CatanPolicy
    league_pool_dicts: list = torch.load(io.BytesIO(pool_bytes), weights_only=False)
    snapshot_policies: list = []
    for snap_dict in league_pool_dicts:
        snap_p = CatanPolicy(
            obs_size=env.obs_size(),
            action_size=env.action_size(),
            hidden=hidden_size,
        )
        snap_p.load_state_dict(snap_dict)
        snap_p.eval()
        snapshot_policies.append(snap_p)

    obs_list:   list = []
    masks_list: list = []
    actions:    list = []
    log_probs:  list = []
    values:     list = []
    rewards:    list = []
    dones:      list = []
    wins:       dict = {}
    last_step_idx: dict = {}   # pid -> last buffer index in the current game

    obs_np, mask_np = env.reset()

    for _ in range(n_steps):
        pid = env.current_player

        # Opponent seat — may use a frozen league snapshot
        if pid != 0 and snapshot_policies and pyrandom.random() < league_prob:
            snap_p = pyrandom.choice(snapshot_policies)
            obs_t  = torch.from_numpy(obs_np).float()
            mask_t = torch.from_numpy(mask_np)
            with torch.no_grad():
                logits, _ = snap_p(obs_t)
                logits[~mask_t] = -1e9
                action_idx = torch.multinomial(torch.softmax(logits, -1), 1).item()
            _, done = env.step(action_idx)
            obs_np, mask_np = env.reset() if done else env.observe()
            continue   # snapshot steps don't go in training buffer

        # Live policy acts
        obs_t  = torch.from_numpy(obs_np).float()
        mask_t = torch.from_numpy(mask_np)

        with torch.no_grad():
            logits, value = policy(obs_t)
            logits[~mask_t] = -1e9
            probs  = torch.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

        rew, done = env.step(action.item())

        obs_list.append(obs_np)
        masks_list.append(mask_np)
        actions.append(action.item())
        log_probs.append(dist.log_prob(action).item())
        values.append(value.squeeze().item())
        rewards.append(float(rew[pid]))
        dones.append(float(done))
        last_step_idx[pid] = len(rewards) - 1

        if done:
            if env.winner is not None:
                wins[env.winner] = wins.get(env.winner, 0) + 1
                loss_pen = reward_config.get("loss_penalty", 5.0)
                for lp, idx in last_step_idx.items():
                    if lp != env.winner:
                        rewards[idx] -= loss_pen
            last_step_idx.clear()

        obs_np, mask_np = env.reset() if done else env.observe()

    obs_size   = env.obs_size()
    act_size   = env.action_size()
    return {
        "obs":       np.array(obs_list,   dtype=np.float32) if obs_list   else np.empty((0, obs_size),  dtype=np.float32),
        "masks":     np.array(masks_list, dtype=bool)       if masks_list else np.empty((0, act_size),  dtype=bool),
        "actions":   np.array(actions,    dtype=np.int64),
        "log_probs": np.array(log_probs,  dtype=np.float32),
        "values":    np.array(values,     dtype=np.float32),
        "rewards":   rewards,
        "dones":     dones,
        "wins":      wins,
    }


# ── Main: setup + training loop ───────────────────────────────────────────────
# Must be guarded for Windows multiprocessing (spawn method).

if __name__ == "__main__":
    import numpy as np

    env           = CatanEnv(num_players=NUM_PLAYERS, reward_shaping=True, **REWARD_CONFIG)
    policy, optim = make_policy(env)

    resume_path    = _latest_checkpoint_path(CKPT_PHASE3)
    save_path      = _next_checkpoint_path(CKPT_PHASE3)
    save_pool_path = _next_checkpoint_path(CKPT_LEAGUE)

    # Priority: resume mid-phase > start from phase 2 > phase 1 > scratch
    if resume_path is not None:
        try:
            policy.load_state_dict(torch.load(resume_path, weights_only=True))
            print(f"Resumed policy from {resume_path.name}")
        except RuntimeError as e:
            print(f"WARNING: Could not load checkpoint {resume_path.name}: {e}")
            print("Starting from scratch (checkpoint architecture mismatch).")
    else:
        phase2_path = _latest_checkpoint_path(CKPT_PHASE2)
        phase1_path = _latest_checkpoint_path(CKPT_PHASE1)
        if phase2_path is not None:
            try:
                policy.load_state_dict(torch.load(phase2_path, weights_only=True))
                print(f"Loaded Phase 2 weights from {phase2_path.name}")
            except RuntimeError as e:
                print(f"WARNING: Could not load Phase 2 checkpoint: {e}")
                print("Starting from random weights (checkpoint architecture mismatch).")
        elif phase1_path is not None:
            try:
                policy.load_state_dict(torch.load(phase1_path, weights_only=True))
                print(f"WARNING: no {Path(CKPT_PHASE2).stem} checkpoint found. "
                      f"Loaded Phase 1 weights from {phase1_path.name}.")
            except RuntimeError as e:
                print(f"WARNING: Could not load Phase 1 checkpoint: {e}")
                print("Starting from random weights (checkpoint architecture mismatch).")
        else:
            print("WARNING: No prior checkpoint found. Starting from random weights.")

    # Resume the league pool if one was already started
    league_pool: list[dict] = []
    league_pool_path = _latest_checkpoint_path(CKPT_LEAGUE)
    if league_pool_path is not None:
        league_pool = torch.load(league_pool_path, weights_only=False)
        print(f"Loaded league pool ({len(league_pool)} snapshots) from {league_pool_path.name}")

    n_per_worker = N_STEPS // N_WORKERS
    print(f"\nPhase 3: league play for {NUM_UPDATES} updates.")
    print(f"Snapshot every {SNAPSHOT_EVERY} updates  |  league opponent prob {LEAGUE_PROB:.0%}")
    print(f"{N_WORKERS} workers × {n_per_worker} steps = {N_STEPS} steps/update\n")

    win_counts: dict = defaultdict(int)

    with mp.Pool(N_WORKERS) as pool:
        for update in range(1, NUM_UPDATES + 1):

            # Snapshot the policy into the league pool periodically
            if update % SNAPSHOT_EVERY == 0:
                league_pool.append(copy.deepcopy(policy.state_dict()))
                torch.save(league_pool, save_pool_path)
                print(f"  [League] Snapshot added — pool size: {len(league_pool)}")

            # Serialize live policy + league pool once per update
            w_io = io.BytesIO()
            torch.save(policy.state_dict(), w_io)
            weights_bytes = w_io.getvalue()

            p_io = io.BytesIO()
            torch.save(league_pool, p_io)
            pool_bytes = p_io.getvalue()

            worker_args = [
                (weights_bytes, pool_bytes, n_per_worker,
                 NUM_PLAYERS, REWARD_CONFIG, LEAGUE_PROB, HIDDEN_SIZE)
            ] * N_WORKERS

            results = pool.map(_worker_collect, worker_args)

            # Concatenate only non-empty worker buffers
            non_empty = [r for r in results if len(r["obs"]) > 0]
            if not non_empty:
                continue   # all steps were league snapshots (very unlikely)

            all_obs       = np.concatenate([r["obs"]       for r in non_empty], axis=0)
            all_masks     = np.concatenate([r["masks"]     for r in non_empty], axis=0)
            all_actions   = np.concatenate([r["actions"]   for r in non_empty])
            all_log_probs = np.concatenate([r["log_probs"] for r in non_empty])
            all_values    = np.concatenate([r["values"]    for r in non_empty])
            all_rewards   = sum([r["rewards"] for r in non_empty], [])
            all_dones     = sum([r["dones"]   for r in non_empty], [])

            for r in results:
                for pid, count in r["wins"].items():
                    win_counts[pid] += count

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
                total    = sum(win_counts.values()) or 1
                rates    = {p: win_counts[p] / total for p in range(NUM_PLAYERS)}
                rate_str = " | ".join(
                    f"p{p} {rates.get(p, 0):.2f}"
                    for p in range(NUM_PLAYERS)
                )
                pool_sz = len(league_pool)
                print(f"Update {update:5d}/{NUM_UPDATES} | pool {pool_sz:3d} | "
                      f"loss {total_loss:9.4f} | {rate_str}")
                win_counts.clear()

    torch.save(policy.state_dict(), save_path)
    torch.save(league_pool, save_pool_path)
    print(f"\nPhase 3 complete.")
    print(f"Saved policy  → {save_path.name}")
    print(f"Saved pool    → {save_pool_path.name}  ({len(league_pool)} snapshots)")
