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
from policy import CatanPolicy

# Edit these here and every phase file picks up the change automatically.

NUM_PLAYERS = 4
HIDDEN_SIZE = 256
LR          = 1e-4
GAMMA       = 0.99    
GAE_LAMBDA  = 0.95    
CLIP_EPS    = 0.2     
VF_COEF     = 0.5     
ENT_COEF    = 0.01    
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
    advantages = []
    gae        = 0.0
    next_val   = 0.0

    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta    = r + gamma * next_val * (1.0 - d) - v.item()
        gae      = delta + gamma * lam * (1.0 - d) * gae
        advantages.insert(0, gae)
        next_val = v.item()

    adv_t = torch.tensor(advantages, dtype=torch.float32)
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
    obs     = torch.stack(buf["obs"])
    masks   = torch.stack(buf["masks"])
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
