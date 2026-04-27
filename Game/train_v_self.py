"""
train_phase2.py — Phase 2: Self-play training.

All 4 seats use the same shared policy.  The network learns to compete
against a competent opponent rather than exploiting random mistakes.

Input  : phase1_policy.pt  (written by train_phase1.py)
           — if not found, starts from random weights with a warning.
         phase2_policy.pt  (optional — resumes mid-phase if it exists)
Output : phase2_policy.pt  (pass this to train_phase3.py to continue)

Run from the Game/ directory:
    python train_phase2.py
"""

import sys, os
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from collections import defaultdict

from catan_env     import CatanEnv
from ppo_utils     import (
    NUM_PLAYERS, N_STEPS, LOG_EVERY,
    GAMMA, GAE_LAMBDA,
    CKPT_PHASE1, CKPT_PHASE2,
    make_policy, compute_gae, ppo_update,
)

# ── Phase 2 specific settings ────────────────────────────────────────────────
NUM_UPDATES = 500   # how many PPO updates to run in this phase
REWARD_CONFIG = {
    "public_vp_reward": 0.3,
    "road_reward": 0.05,
    "buy_dev_reward": 0.10,
    "win_reward": 5.0,
    "loss_penalty": 5.0,
    "setup_settle_reward": 0.5,
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

resume_path = _latest_checkpoint_path(CKPT_PHASE2)
save_path = _next_checkpoint_path(CKPT_PHASE2)

# Priority: resume mid-phase > start from phase 1 > start from scratch
if resume_path is not None:
    policy.load_state_dict(torch.load(resume_path))
    print(f"Resumed from {resume_path.name}")
else:
    phase1_path = _latest_checkpoint_path(CKPT_PHASE1)
    if phase1_path is not None:
        policy.load_state_dict(torch.load(phase1_path))
        print(f"Loaded Phase 1 weights from {phase1_path.name}")
    else:
        print(f"WARNING: no {Path(CKPT_PHASE1).stem} checkpoint found. Run train_phase1.py first for best results.")
        print("Starting Phase 2 from random weights.\n")

print(f"Phase 2: full self-play for {NUM_UPDATES} updates.\n")


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(n_steps: int) -> tuple[dict, dict[int, int]]:
    """
    All 4 seats use the live policy.  Every step is stored in the buffer
    so the policy learns from all player perspectives simultaneously.
    """
    buf = {"obs": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    episode_wins: dict[int, int] = defaultdict(int)
    obs, mask = env.reset()

    for _ in range(n_steps):
        pid    = env.current_player
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

        if done and env.winner is not None:
            episode_wins[env.winner] += 1

        obs, mask = env.reset() if done else env.observe()

    return buf, episode_wins


# ── Training loop ─────────────────────────────────────────────────────────────

win_counts: dict = defaultdict(int)

for update in range(1, NUM_UPDATES + 1):
    buf, episode_wins = collect_rollout(N_STEPS)
    adv, ret   = compute_gae(buf["rewards"], buf["values"], buf["dones"])
    total_loss = ppo_update(buf, adv, ret, policy, optim)

    for pid, count in episode_wins.items():
        win_counts[pid] += count

    if update % LOG_EVERY == 0:
        games    = sum(win_counts.values())
        total    = games or 1
        rates    = {p: win_counts[p] / total for p in range(NUM_PLAYERS)}
        rate_str = " | ".join(
            f"p{p} {rates.get(p, 0):.2f}"
            for p in range(NUM_PLAYERS)
        )
        print(f"Update {update:4d}/{NUM_UPDATES} | loss {total_loss:9.4f} | games {games:3d} | {rate_str}")
        win_counts.clear()


torch.save(policy.state_dict(), save_path)
print(f"\nPhase 2 complete. Saved {save_path.name}")
print(f"Next step: run  python train_phase3.py")
