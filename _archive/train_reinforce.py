import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from catan_env import CatanEnv
from policy import CatanPolicy

NUM_PLAYERS   = 4
HIDDEN_SIZE   = 256
LR            = 3e-4
GAMMA         = 0.99     
ENTROPY_COEF  = 0.01
NUM_EPISODES  = 2000
LOG_EVERY     = 50

# Setup
env    = CatanEnv(num_players=NUM_PLAYERS, reward_shaping=True)
policy = CatanPolicy(obs_size=env.obs_size(), action_size=env.action_size(),
                     hidden=HIDDEN_SIZE)
optim  = torch.optim.Adam(policy.parameters(), lr=LR)


def run_episode():
    """Play one game. Returns per-player trajectory lists."""
    obs, mask = env.reset()

    # Each player accumulates their own trajectory
    trajs = defaultdict(lambda: {"log_probs": [], "rewards": [], "entropies": []})

    while not env.done:
        pid    = env.current_player
        obs_t  = torch.tensor(obs,  dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.bool)

        logits, _value = policy(obs_t)
        masked_logits  = logits.clone()
        masked_logits[~mask_t] = -1e9
        probs  = F.softmax(masked_logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()

        trajs[pid]["log_probs"].append(dist.log_prob(action))
        trajs[pid]["entropies"].append(dist.entropy())

        rewards, done = env.step(action.item())
        trajs[pid]["rewards"].append(rewards[pid])

        if not done:
            obs, mask = env.observe()

    return trajs


def compute_returns(rewards, gamma):
    """Discounted returns from a list of per-step rewards."""
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalise to reduce variance
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# Training loop
win_counts = defaultdict(int)

for episode in range(1, NUM_EPISODES + 1):
    trajs = run_episode()

    loss = torch.tensor(0.0)
    for pid, traj in trajs.items():
        if not traj["log_probs"]:
            continue
        returns    = compute_returns(traj["rewards"], GAMMA)
        log_probs  = torch.stack(traj["log_probs"])
        entropies  = torch.stack(traj["entropies"])
        pg_loss    = -(log_probs * returns).mean()
        entropy_loss = -ENTROPY_COEF * entropies.mean()
        loss       = loss + pg_loss + entropy_loss

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optim.step()

    if env.winner is not None:
        win_counts[env.winner] += 1

    if episode % LOG_EVERY == 0:
        total = sum(win_counts.values())
        rates = {p: win_counts[p] / total for p in range(NUM_PLAYERS)}
        print(f"Episode {episode:5d} | loss {loss.item():7.4f} | "
              f"win rates {[f'{rates.get(p,0):.2f}' for p in range(NUM_PLAYERS)]}")
        win_counts.clear()

# Save the trained weights
torch.save(policy.state_dict(), "catan_policy.pt")
print("Saved catan_policy.pt")