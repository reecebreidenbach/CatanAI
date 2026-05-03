"""
ppo_utils.py — Shared hyperparameters and functions used by all three phase files.

Imported by:
    train_phase1.py  (curriculum)
    train_phase2.py  (self-play)
    train_phase3.py  (league play)
    train_ppo.py     (combined all-in-one)
"""

import torch
import torch.nn.functional as F
import numpy as np
from policy import CatanPolicy
from catan_env import (
    _ACT_ROLL, _ACT_END, _ACT_SETTLE, _ACT_ROAD, _ACT_CITY,
    _ACT_ROBBER, _ACT_TRADE, _ACT_BUY, _ACT_KNIGHT,
    _ACT_MONOPOLY, _ACT_YOP, _ACT_ROAD_BUILDING,
)

# Edit these here and every phase file picks up the change automatically.

NUM_PLAYERS = 4
HIDDEN_SIZE = 256
LR          = 1e-4
GAMMA       = 0.99    
GAE_LAMBDA  = 0.95    
CLIP_EPS    = 0.2     
VF_COEF     = 0.5     
ENT_COEF    = 0.03    
N_STEPS     = 4096    # large enough to contain several complete Catan games
N_EPOCHS    = 4       
BATCH_SIZE  = 64      
LOG_EVERY   = 10      

# ── Checkpoint filenames ─────────────────────────────────────────────────────
# Phase 1 writes CKPT_PHASE1.  Phase 2 reads it and writes CKPT_PHASE2.
# Phase 3 reads CKPT_PHASE2 and writes CKPT_PHASE3 + CKPT_LEAGUE.

CKPT_PHASE1 = "phase1_policy.pt"   # output of train_phase1.py
CKPT_PHASE2 = "phase2_policy.pt"   # output of train_phase2.py
CKPT_PHASE3 = "phase3_policy.pt"   # output of train_phase3.py
CKPT_LEAGUE = "league_pool.pt"     # list of frozen snapshots (phase 3)


# ── Action-type classifier (shared across all phase files) ───────────────────

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


# ── Per-phase reward configs ──────────────────────────────────────────────────
# Phase 1 uses stronger shaping to bootstrap basic game play quickly.
# Phase 2/3 anneal shaping down toward pure win/loss so the policy doesn't
# overfit to the proxy signals.
#
# The env already applies win_reward and loss_penalty in step() at game end.
# Rollout code must NOT subtract loss_penalty a second time.

REWARD_CONFIG_PHASE1 = {
    # Terminal
    "win_reward":          5.0,
    "loss_penalty":        7.0,   # strong — losing to randoms/greedy should hurt
    # Dense shaping (higher in Phase 1 to guide early learning)
    "public_vp_reward":    0.5,
    "road_reward":         0.08,
    "buy_dev_reward":      0.10,
    "setup_settle_reward": 5.0,   # Phase1: max spot ~4.3, worst ~1.7 — delta 2.6 > game noise
    "robber_block_reward": 0.1,
    "monopoly_reward":     0.3,
    "yop_build_reward":    0.15,
    "city_pip_reward":     0.15,
    "settlement_prod_reward": 0.12,
    "city_prod_reward":    0.12,
    "road_waste_penalty":  0.05,
    "near_settlement_road_penalty": 0.0,
    "maritime_trade_penalty": 0.0,
    "empty_trade_penalty": 0.0,
    "robber_leader_bonus": 0.1,
}

REWARD_CONFIG_PHASE2 = {
    # Terminal (same magnitude — win still outweighs loss)
    "win_reward":          5.0,
    "loss_penalty":        2.0,   # softer — all seats learn simultaneously
    # Dense shaping (reduced — rely more on win/loss signal in self-play)
    "public_vp_reward":    0.3,
    "road_reward":         0.05,
    "buy_dev_reward":      0.07,
    "setup_settle_reward": 4.0,   # Phase2: max spot ~3.5, worst ~1.3 — habit locks in
    "robber_block_reward": 0.07,
    "monopoly_reward":     0.2,
    "yop_build_reward":    0.10,
    "city_pip_reward":     0.10,
    "settlement_prod_reward": 0.10,
    "city_prod_reward":    0.14,
    "road_waste_penalty":  0.04,
    "near_settlement_road_penalty": 0.0,
    "maritime_trade_penalty": 0.01,
    "empty_trade_penalty": 0.02,
    "robber_leader_bonus": 0.07,
}

REWARD_CONFIG_PHASE3 = {
    # Terminal
    "win_reward":          5.0,
    "loss_penalty":        3.0,   # moderate — league opponents are past selves
    # Dense shaping (minimal — close to pure win/loss by league stage)
    "public_vp_reward":    0.2,
    "road_reward":         0.02,
    "buy_dev_reward":      0.05,
    "setup_settle_reward": 3.0,   # Phase3: max spot ~2.6, worst ~1.0 — win signal still dominates
    "robber_block_reward": 0.05,
    "monopoly_reward":     0.15,
    "yop_build_reward":    0.07,
    "city_pip_reward":     0.14,
    "settlement_prod_reward": 0.14,
    "city_prod_reward":    0.20,
    "road_waste_penalty":  0.08,
    "near_settlement_road_penalty": 0.10,
    "setup_road_reward":   0.75,
    "expansion_stall_penalty": 0.10,
    "opening_strategy_bonus": 0.08,
    "maritime_trade_penalty": 0.02,
    "empty_trade_penalty": 0.05,
    "robber_leader_bonus": 0.05,
}


# ── Factory helpers ──────────────────────────────────────────────────────────

def make_policy(env) -> tuple:
    """
    Create a fresh CatanPolicy + Adam optimiser pair.

    Returns
    -------
    (policy, optim)
    """
    p = CatanPolicy(
        obs_size=env.obs_size(),
        action_size=env.action_size(),
        hidden=HIDDEN_SIZE,
    )
    o = torch.optim.Adam(p.parameters(), lr=LR)
    return p, o


def league_action(obs, mask, state_dict: dict, env) -> int:
    """
    Run a forward pass through a frozen league snapshot and return an action.
    Used so opponent seats can play as older versions of the policy.

    Parameters
    ----------
    obs        : current observation (numpy array)
    mask       : legal action mask (numpy bool array)
    state_dict : weights of the frozen snapshot
    env        : CatanEnv instance (needed for obs/action sizes)
    """
    frozen = CatanPolicy(
        obs_size=env.obs_size(),
        action_size=env.action_size(),
        hidden=HIDDEN_SIZE,
    )
    frozen.load_state_dict(state_dict)
    frozen.eval()
    with torch.no_grad():
        obs_t  = torch.tensor(obs,  dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        logits, _ = frozen(obs_t)
        logits[~mask_t] = -1e9
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()


def compute_gae(
    rewards: list,
    values:  list,
    dones:   list,
    gamma:   float = GAMMA,
    lam:     float = GAE_LAMBDA,
) -> tuple:
    """
    Generalized Advantage Estimation (GAE).

    Advantages measure how much better an action was compared to the value
    baseline.  The lam parameter smooths across time:
      lam = 1.0  →  like raw discounted returns  (high variance)
      lam = 0.0  →  like one-step TD error       (high bias)
      lam = 0.95 →  standard sweet spot

    Returns
    -------
    advantages : float32 tensor, shape (n,)
    returns    : float32 tensor, shape (n,)  — value prediction targets
    """
    n          = len(rewards)
    advantages = np.empty(n, dtype=np.float32)
    val_arr    = np.array([v.item() for v in values], dtype=np.float32)
    rew_arr    = np.array(rewards, dtype=np.float32)
    don_arr    = np.array(dones,   dtype=np.float32)

    gae      = 0.0
    next_val = 0.0
    for i in range(n - 1, -1, -1):
        delta       = rew_arr[i] + gamma * next_val * (1.0 - don_arr[i]) - val_arr[i]
        gae         = delta + gamma * lam * (1.0 - don_arr[i]) * gae
        advantages[i] = gae
        next_val    = val_arr[i]

    adv_t = torch.from_numpy(advantages)
    ret_t = adv_t + torch.stack(values)
    return adv_t, ret_t


# ── PPO update ───────────────────────────────────────────────────────────────

def ppo_update(buf: dict, advantages: torch.Tensor, returns: torch.Tensor,
               policy, optim) -> float:
    """
    Run N_EPOCHS × mini-batch PPO updates on one collected rollout.

    The clipped surrogate objective stops the policy from changing too
    much in a single update step — the key stability improvement over
    plain REINFORCE.

    Parameters
    ----------
    buf        : rollout buffer from collect_rollout()
    advantages : GAE advantages, shape (n,)
    returns    : value targets,  shape (n,)
    policy     : the CatanPolicy being trained
    optim      : its Adam optimiser

    Returns
    -------
    total_loss : summed scalar loss over all mini-batch steps (for logging)
    """
    # obs and masks may be a mix of per-episode 2-D tensors (from the fast
    # numpy-batch path) and individual 1-D tensors (from older callers).
    # cat handles both cases uniformly.
    obs     = torch.cat([t if t.dim() == 2 else t.unsqueeze(0) for t in buf["obs"]])
    masks   = torch.cat([t if t.dim() == 2 else t.unsqueeze(0) for t in buf["masks"]])
    actions = torch.stack(buf["actions"])
    old_lp  = torch.stack(buf["log_probs"]).detach()
    n       = len(obs)
    total_loss = 0.0

    for _ in range(N_EPOCHS):
        idxs = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            b = idxs[start : start + BATCH_SIZE]

            logits, values = policy(obs[b])
            masked_logits = logits.masked_fill(~masks[b], -1e9)
            dist    = torch.distributions.Categorical(logits=masked_logits)
            new_lp  = dist.log_prob(actions[b])
            entropy = dist.entropy().mean()

            ratio  = (new_lp - old_lp[b]).exp()
            adv_b  = advantages[b]
            adv_b  = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)
            surr1  = ratio * adv_b
            surr2  = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b

            pg_loss = -torch.min(surr1, surr2).mean()
            vf_loss = F.mse_loss(values.squeeze(-1), returns[b])
            loss    = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optim.step()
            total_loss += loss.item()

    return total_loss
