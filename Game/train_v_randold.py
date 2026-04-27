"""
train_phase3.py — Phase 3: League play training.

Every SNAPSHOT_EVERY updates, a frozen copy (snapshot) of the current
policy is added to the league pool.  Opponent seats randomly draw from
the pool (old versions) or use the live policy, so the agent must handle
both old and new playstyles simultaneously.

This prevents strategy collapse — without a pool, the policy can overfit
to beating its current self while forgetting how to beat earlier strategies.

Input  : phase2_policy.pt  (written by train_phase2.py)
           — if not found, starts from phase1_policy.pt, then random weights.
         phase3_policy.pt  (optional — resumes mid-phase if it exists)
         league_pool.pt    (optional — resumes the snapshot pool if it exists)
Output : phase3_policy.pt
         league_pool.pt

Run from the Game/ directory:
    python train_phase3.py
"""

import sys, os, copy, random as pyrandom
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from collections import defaultdict

from catan_env     import CatanEnv
from ppo_utils     import (
    NUM_PLAYERS, HIDDEN_SIZE, N_STEPS, LOG_EVERY,
    GAMMA, GAE_LAMBDA,
    CKPT_PHASE1, CKPT_PHASE2, CKPT_PHASE3, CKPT_LEAGUE,
    make_policy, league_action, compute_gae, ppo_update,
)

# ── Phase 3 specific settings ─────────────────────────────────────────────────
NUM_UPDATES    = 2000   # how many PPO updates to run in this phase
SNAPSHOT_EVERY = 100    # freeze a snapshot into the pool every N updates
LEAGUE_PROB    = 0.5    # chance any opponent seat draws from the pool
                        # (vs. using the current live policy)
REWARD_CONFIG = {
    "public_vp_reward": 0.3,
    "road_reward": 0.05,
    "buy_dev_reward": 0.10,
    "win_reward": 5.0,
    "loss_penalty": 5.0,
    "setup_settle_reward": 0.5,
    "robber_block_reward": 0.1,
    "monopoly_reward": 0.3,
    "yop_build_reward": 0.15,
}


def _checkpoint_dir() -> Path:
    return Path(__file__).resolve().parent


def _checkpoint_base(path_str: str) -> tuple[str, str]:
    ckpt_path = Path(path_str)
    return ckpt_path.stem, ckpt_path.suffix or ".pt"


def _latest_checkpoint_path(path_str: str) -> Path | None:
    base_stem, suffix = _checkpoint_base(path_str)
    ckpt_dir = _checkpoint_dir()
    candidates = sorted(ckpt_dir.glob(f"{base_stem}*{suffix}"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _next_checkpoint_path(path_str: str) -> Path:
    base_stem, suffix = _checkpoint_base(path_str)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _checkpoint_dir() / f"{base_stem}_{timestamp}{suffix}"

# ── Setup ────────────────────────────────────────────────────────────────────
env           = CatanEnv(num_players=NUM_PLAYERS, reward_shaping=True, **REWARD_CONFIG)
policy, optim = make_policy(env)

resume_path = _latest_checkpoint_path(CKPT_PHASE3)
save_path = _next_checkpoint_path(CKPT_PHASE3)
save_pool_path = _next_checkpoint_path(CKPT_LEAGUE)

# Priority: resume mid-phase > start from phase 2 > phase 1 > scratch
if resume_path is not None:
    policy.load_state_dict(torch.load(resume_path))
    print(f"Resumed policy from {resume_path.name}")
else:
    phase2_path = _latest_checkpoint_path(CKPT_PHASE2)
    phase1_path = _latest_checkpoint_path(CKPT_PHASE1)
    if phase2_path is not None:
        policy.load_state_dict(torch.load(phase2_path))
        print(f"Loaded Phase 2 weights from {phase2_path.name}")
    elif phase1_path is not None:
        policy.load_state_dict(torch.load(phase1_path))
        print(f"WARNING: no {Path(CKPT_PHASE2).stem} checkpoint found. Loaded Phase 1 weights from {phase1_path.name}.")
    else:
        print(f"WARNING: No prior checkpoint found. Starting from random weights.")

# Resume the league pool if one was already started
league_pool: list[dict] = []
league_pool_path = _latest_checkpoint_path(CKPT_LEAGUE)
if league_pool_path is not None:
    league_pool = torch.load(league_pool_path)
    print(f"Loaded league pool ({len(league_pool)} snapshots) from {league_pool_path.name}")

print(f"\nPhase 3: league play for {NUM_UPDATES} updates.")
print(f"Snapshot every {SNAPSHOT_EVERY} updates  |  league opponent prob {LEAGUE_PROB:.0%}\n")


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(n_steps: int) -> dict:
    """
    Collect n_steps steps.

    Player 0 always uses the live policy (always trains).
    Opponent seats (1–3) each flip a coin:
      - LEAGUE_PROB  → draw a random frozen snapshot from the pool
      - 1-LEAGUE_PROB → use the live policy

    Steps taken by a league snapshot are NOT stored in the gradient buffer
    (we don't want to train the policy on its own old decisions).
    """
    buf = {"obs": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    obs, mask = env.reset()

    for _ in range(n_steps):
        pid = env.current_player

        # Opponent seat — may use a league snapshot
        if pid != 0 and league_pool and pyrandom.random() < LEAGUE_PROB:
            snap       = pyrandom.choice(league_pool)
            action_idx = league_action(obs, mask, snap, env)
            _, done    = env.step(action_idx)
            obs, mask  = env.reset() if done else env.observe()
            continue   # snapshot steps don't go in the training buffer

        # Live policy acts
        obs_t  = torch.tensor(obs,  dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)

        with torch.no_grad():
            logits, value = policy(obs_t)
            logits[~mask_t] = -1e9
            probs  = torch.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

        rewards, done = env.step(action.item())
        reward = rewards[pid]

        buf["obs"].append(obs_t)
        buf["masks"].append(mask_t.clone())
        buf["actions"].append(action)
        buf["log_probs"].append(dist.log_prob(action))
        buf["values"].append(value.squeeze())
        buf["rewards"].append(reward)
        buf["dones"].append(float(done))

        obs, mask = env.reset() if done else env.observe()

    return buf


# ── Training loop ─────────────────────────────────────────────────────────────

win_counts: dict = defaultdict(int)

for update in range(1, NUM_UPDATES + 1):

    # Snapshot the policy into the league pool periodically
    if update % SNAPSHOT_EVERY == 0:
        league_pool.append(copy.deepcopy(policy.state_dict()))
        torch.save(league_pool, save_pool_path)   # persist the pool to disk for this run
        print(f"  [League] Snapshot added — pool size: {len(league_pool)}")

    buf = collect_rollout(N_STEPS)

    if not buf["obs"]:
        continue   # all steps were league snapshots (very unlikely but safe)

    adv, ret   = compute_gae(buf["rewards"], buf["values"], buf["dones"])
    total_loss = ppo_update(buf, adv, ret, policy, optim)

    if env.winner is not None:
        win_counts[env.winner] += 1

    if update % LOG_EVERY == 0:
        total    = sum(win_counts.values()) or 1
        rates    = {p: win_counts[p] / total for p in range(NUM_PLAYERS)}
        rate_str = " | ".join(
            f"p{p} {rates.get(p, 0):.2f}"
            for p in range(NUM_PLAYERS)
        )
        pool_sz  = len(league_pool)
        print(f"Update {update:5d}/{NUM_UPDATES} | pool {pool_sz:3d} | loss {total_loss:9.4f} | {rate_str}")
        win_counts.clear()

torch.save(policy.state_dict(), save_path)
torch.save(league_pool, save_pool_path)
print(f"\nPhase 3 complete.")
print(f"Saved policy  → {save_path.name}")
print(f"Saved pool    → {save_pool_path.name}  ({len(league_pool)} snapshots)")
