"""
train_phase1.py — Phase 1: Curriculum training (learning agent vs random opponents).

One seat per game is chosen uniformly at random to use the learning policy.
The remaining seats use RandomAgents that pick actions uniformly at random.
This lets the network learn basic Catan moves without overfitting to a fixed
board position in turn order.

Input  : phase1_policy.pt  (optional — resumes if it already exists)
Output : phase1_policy.pt  (pass this to train_phase2.py to continue)

Run from the Game/ directory:
    python train_phase1.py
"""

import sys, os, random
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from collections import defaultdict

from catan_env     import CatanEnv, RandomAgent, GreedyVPAgent, _ACT_ROLL, _ACT_END, _ACT_SETTLE, _ACT_ROAD, _ACT_CITY, _ACT_ROBBER, _ACT_TRADE, _ACT_BUY, _ACT_KNIGHT, _ACT_MONOPOLY, _ACT_YOP, _ACT_ROAD_BUILDING
from ppo_utils     import (
    NUM_PLAYERS, HIDDEN_SIZE, N_STEPS, LOG_EVERY,
    GAMMA, GAE_LAMBDA,
    CKPT_PHASE1,
    REWARD_CONFIG_PHASE1 as REWARD_CONFIG,
    make_policy, compute_gae, ppo_update,
)
import ppo_utils as _ppo_utils
_ppo_utils.ENT_COEF = 0.05   # Phase 1 needs stronger entropy push; single-seat training collapses fast

# ── Phase 1 specific settings ────────────────────────────────────────────────
NUM_UPDATES          = 300   # how many PPO updates to run in this phase
N_EPISODES_PER_UPDATE = 4    # complete games collected before each update
EPISODE_TIMEOUT      = 3000  # steps per game safety limit (no winner declared)
# OPPONENT_POOL: mix of random and greedy agents for curriculum variety.
# 50% greedy makes opponents meaningfully stronger than pure random.
OPPONENT_POOL_MIX = 0.5   # fraction of opponent seats that use GreedyVPAgent


def _checkpoint_dir() -> Path:
    return Path(__file__).resolve().parent


def _checkpoint_base() -> tuple[str, str]:
    ckpt_path = Path(CKPT_PHASE1)
    return ckpt_path.stem, ckpt_path.suffix or ".pt"


def _latest_checkpoint_path() -> Path | None:
    base_stem, suffix = _checkpoint_base()
    ckpt_dir = _checkpoint_dir()
    candidates = sorted(ckpt_dir.glob(f"{base_stem}*{suffix}"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _next_checkpoint_path() -> Path:
    base_stem, suffix = _checkpoint_base()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _checkpoint_dir() / f"{base_stem}_{timestamp}{suffix}"

# ── Setup ────────────────────────────────────────────────────────────────────
env            = CatanEnv(num_players=NUM_PLAYERS, reward_shaping=True, **REWARD_CONFIG)
policy, optim  = make_policy(env)
# Mixed opponent pool: some greedy, some random, decided per episode.
_greedy_agent  = GreedyVPAgent(env)
_random_agent  = RandomAgent()

resume_path = _latest_checkpoint_path()
save_path = _next_checkpoint_path()

if resume_path is not None:
    try:
        policy.load_state_dict(torch.load(resume_path))
        print(f"Resumed from {resume_path.name}")
    except RuntimeError as e:
        print(f"WARNING: Could not load checkpoint {resume_path.name}: {e}")
        print("Starting Phase 1 from scratch (checkpoint architecture mismatch).")
else:
    print("Starting Phase 1 from scratch.")

print(f"Phase 1: one random seat learns vs {NUM_PLAYERS - 1} RandomAgents for {NUM_UPDATES} updates.\n")


# ── Rollout collection ────────────────────────────────────────────────────────

def _act_type(idx: int) -> str:
    if idx == _ACT_ROLL:                            return "roll"
    if idx == _ACT_END:                             return "end"
    if _ACT_SETTLE <= idx < _ACT_ROAD:              return "settle"
    if _ACT_ROAD   <= idx < _ACT_CITY:              return "road"
    if _ACT_CITY   <= idx < _ACT_ROBBER:            return "city"
    if _ACT_ROBBER <= idx < _ACT_END:               return "robber"
    if _ACT_TRADE  <= idx < _ACT_BUY:               return "trade"
    if idx == _ACT_BUY:                             return "buydev"
    if idx == _ACT_KNIGHT:                          return "knight"
    if _ACT_MONOPOLY <= idx < _ACT_YOP:             return "monopoly"
    if _ACT_YOP <= idx < _ACT_ROAD_BUILDING:        return "yop"
    if idx == _ACT_ROAD_BUILDING:                   return "roadbuild"
    return "other"


def collect_rollout(n_episodes: int) -> tuple:
    """
    Collect n_episodes complete games where one randomly selected seat uses
    the learning policy and contributes to the gradient buffer. All other
    seats act via RandomAgent.

    Each episode runs until the game ends naturally (a player reaches 10 VP).
    If a game exceeds EPISODE_TIMEOUT steps it is abandoned with no winner
    recorded, so the agent cannot learn to exploit an artificial time limit.

    Returns (buf, episode_wins, learner_wins, action_counts).
    """
    import numpy as np
    obs_size = env.obs_size()
    buf = {"obs": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    episode_wins: dict = defaultdict(int)
    learner_wins = 0
    action_counts: dict = defaultdict(int)

    for _ in range(n_episodes):
        learner_pid      = random.randrange(NUM_PLAYERS)
        obs, mask       = env.reset()
        ep_steps        = 0
        ep_obs:      list = []
        ep_masks:    list = []
        ep_actions:  list = []
        ep_log_probs:list = []
        ep_values:   list = []
        ep_rewards:  list = []
        ep_dones:    list = []

        while True:
            ep_steps += 1
            if ep_steps > EPISODE_TIMEOUT:
                break

            pid = env.current_player

            if pid != learner_pid:
                opp = _greedy_agent if random.random() < OPPONENT_POOL_MIX else _random_agent
                action_idx = opp.choose(obs, mask)
                _, done    = env.step(action_idx)
                if done:
                    if env.winner is not None:
                        episode_wins[env.winner] += 1
                        learner_wins += int(env.winner == learner_pid)
                    break
                else:
                    obs, mask = env.observe()
                continue

            # Learner seat — live policy, record for training
            obs_t  = torch.from_numpy(obs).float()
            mask_t = torch.from_numpy(mask)

            with torch.no_grad():
                logits, value = policy(obs_t)
                logits[~mask_t] = -1e9
                probs  = torch.softmax(logits, dim=-1)
                dist   = torch.distributions.Categorical(probs)
                action = dist.sample()

            rewards, done = env.step(action.item())
            reward = rewards[learner_pid]
            action_counts[_act_type(action.item())] += 1

            ep_obs.append(obs)
            ep_masks.append(mask)
            ep_actions.append(action)
            ep_log_probs.append(dist.log_prob(action))
            ep_values.append(value.squeeze(-1))
            ep_rewards.append(reward)
            ep_dones.append(float(done))

            if done:
                if env.winner is not None:
                    episode_wins[env.winner] += 1
                    learner_wins += int(env.winner == learner_pid)
                break
            else:
                obs, mask = env.observe()

        if ep_obs:
            # Convert numpy arrays in bulk rather than stacking per-step tensors
            buf["obs"].append(torch.from_numpy(np.array(ep_obs, dtype=np.float32)))
            buf["masks"].append(torch.from_numpy(np.array(ep_masks, dtype=bool)))
            buf["actions"].extend(ep_actions)
            buf["log_probs"].extend(ep_log_probs)
            buf["values"].extend(ep_values)
            buf["rewards"].extend(ep_rewards)
            buf["dones"].extend(ep_dones)

    return buf, episode_wins, learner_wins, action_counts

    return buf, episode_wins, learner_wins, action_counts


# ── Training loop ─────────────────────────────────────────────────────────────

win_counts: dict    = defaultdict(int)
learner_win_total   = 0
games_total         = 0
act_totals: dict    = defaultdict(int)

for update in range(1, NUM_UPDATES + 1):
    buf, episode_wins, learner_wins, action_counts = collect_rollout(N_EPISODES_PER_UPDATE)

    if not buf["obs"]:
        continue   # all steps were random agents (shouldn't happen after first game)

    adv, ret   = compute_gae(buf["rewards"], buf["values"], buf["dones"])
    total_loss = ppo_update(buf, adv, ret, policy, optim)

    for pid, count in episode_wins.items():
        win_counts[pid] += count
    learner_win_total += learner_wins
    games_total       += sum(episode_wins.values())
    for act, count in action_counts.items():
        act_totals[act] += count

    if update % LOG_EVERY == 0:
        games  = sum(win_counts.values())
        total  = games or 1
        learner_rate = learner_win_total / (games_total or 1)
        n_acts = sum(act_totals.values()) or 1
        acts   = "  ".join(f"{k}:{act_totals[k]/n_acts:.2f}" for k in
                           ["roll","end","settle","road","city","trade","robber","buydev"])
        print(f"Update {update:4d}/{NUM_UPDATES} | loss {total_loss:9.4f} | games {games:3d} | learner {learner_rate:.2f} | entropy {torch.mean(-torch.stack(buf['log_probs'])).item():.4f}")
        print(f"  actions: {acts}")
        win_counts.clear()
        learner_win_total = 0
        games_total = 0
        act_totals.clear()

torch.save(policy.state_dict(), save_path)
print(f"\nPhase 1 complete. Saved {save_path.name}")
print(f"Next step: run  python train_phase2.py")
