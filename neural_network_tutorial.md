# Adding a Neural Network to CatanAI

This tutorial walks through adding a neural network policy to the Catan RL environment, starting from scratch.
By the end you will have a working PPO training loop that can train an agent to play Catan.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [How the environment works (quick recap)](#2-how-the-environment-works-quick-recap)
3. [Architecture choices](#3-architecture-choices)
4. [Minimal MLP policy with PyTorch](#4-minimal-mlp-policy-with-pytorch)
5. [Masking illegal actions](#5-masking-illegal-actions)
6. [Self-play training loop (REINFORCE)](#6-self-play-training-loop-reinforce)
7. [Upgrading to PPO](#7-upgrading-to-ppo)
8. [Tracking progress](#8-tracking-progress)
9. [What to try next](#9-what-to-try-next)

---

## 1. Prerequisites

Install the required packages into the project virtual environment:

```bash
# From the project root (CatanAI/)
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install numpy matplotlib
```

> Use the `+cu121` PyTorch variant instead if you have an NVIDIA GPU.

Confirm everything imports:

```python
import torch
import numpy as np
from Game.catan_env import CatanEnv
print(torch.__version__, np.__version__)
```

---

## 2. How the environment works (quick recap)

The key loop is:

```python
env = CatanEnv(num_players=4, reward_shaping=True)
obs, mask = env.reset()   # obs: float32 array of shape (950,)
                           # mask: bool array of shape (298,)

while not env.done:
    obs, mask = env.observe()   # always from current player's view
    action_idx = agent.choose(obs, mask)
    rewards, done = env.step(action_idx)
    # rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    # winner gets +1.0 on the terminal step
```

Key numbers:

| Symbol              | Value   | Meaning                                 |
| ------------------- | ------- | --------------------------------------- |
| `env.obs_size()`    | **950** | Input dimension of the network          |
| `env.action_size()` | **298** | Output dimension (one logit per action) |

The `mask` is a boolean array that is `True` for every action that is legal in the current state.
**You must zero-out illegal logits before sampling** — otherwise the agent will take actions that crash the game engine.

---

## 3. Architecture choices

For a first pass, a simple **Multi-Layer Perceptron (MLP)** is the right choice:

- Fast to train
- No spatial assumptions needed (the board topology is already encoded in the flat vector)
- Easy to debug

Later improvements to consider (not covered here):

- **Graph Neural Network (GNN)** — model the board as a graph of hexes/vertices/edges so the network can generalise across board layouts.
- **Transformer** — attend over the 54 vertices or 19 hexes directly.
- **AlphaZero-style (MCTS + value head)** — use `env.state()` to expand the game tree with a learned value function.

---

## 4. Minimal MLP policy with PyTorch

Create a file `Game/policy.py`:

```python
import torch
import torch.nn as nn

class CatanPolicy(nn.Module):
    """
    Actor-Critic MLP for Catan.

    Input:  obs  — float32 tensor of shape (batch, obs_size)
    Output: logits — raw scores for each action, shape (batch, action_size)
            value  — estimated game value, shape (batch, 1)
    """

    def __init__(self, obs_size: int, action_size: int, hidden: int = 256):
        super().__init__()

        # Shared trunk: compress the 950-dim observation into a rich feature vector
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Policy head: one logit per action (298 total)
        self.policy_head = nn.Linear(hidden, action_size)

        # Value head: scalar estimate of "how likely am I to win from here"
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logits   = self.policy_head(features)   # (batch, 298)
        value    = self.value_head(features)     # (batch, 1)
        return logits, value
```

Test it immediately:

```python
from Game.catan_env import CatanEnv
from Game.policy import CatanPolicy
import torch

env    = CatanEnv(num_players=4)
policy = CatanPolicy(obs_size=env.obs_size(), action_size=env.action_size())

obs, mask = env.reset()
obs_t  = torch.tensor(obs).unsqueeze(0)   # (1, 950)
logits, value = policy(obs_t)

print("logits shape:", logits.shape)   # → torch.Size([1, 298])
print("value shape: ", value.shape)    # → torch.Size([1, 1])
```

---

## 5. Masking illegal actions

The network produces a logit for all 298 actions regardless of whether they are legal.
Before sampling you must set illegal logit scores to a very large negative number so they
get probability ≈ 0 after softmax:

```python
import torch
import torch.nn.functional as F

def masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> int:
    """
    Sample an action from the distribution, with illegal actions masked out.

    logits : (298,)  raw network output
    mask   : (298,)  bool tensor, True = legal
    """
    masked = logits.clone()
    masked[~mask] = -1e9                         # kill illegal actions
    probs  = F.softmax(masked, dim=-1)           # turn into a probability distribution
    action = torch.multinomial(probs, 1).item()  # sample one action
    return action

# Usage inside the game loop:
obs_t  = torch.tensor(obs, dtype=torch.float32)
mask_t = torch.tensor(mask, dtype=torch.bool)
logits, value = policy(obs_t)
action_idx = masked_sample(logits, mask_t)
```

> **Why not argmax?**
> Argmax is greedy and stops exploring. During training you _want_ to occasionally
> try suboptimal actions so the agent can discover better strategies.
> Use `argmax` (greedy) only at evaluation/test time.

---

## 6. Self-play training loop (REINFORCE)

REINFORCE (also called Monte Carlo Policy Gradient) is the simplest working algorithm.
Each episode generates a trajectory; after the game ends, you compute a discounted return
for each step and update the policy to make high-return actions more probable.

Create `Game/train_reinforce.py`:

```python
"""
Minimal REINFORCE self-play training loop for CatanAI.
Run from the Game/ directory:
    python train_reinforce.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from catan_env import CatanEnv
from policy import CatanPolicy

# ── Hyperparameters ─────────────────────────────────────────────────────────
NUM_PLAYERS   = 4
HIDDEN_SIZE   = 256
LR            = 3e-4
GAMMA         = 0.99     # discount factor
ENTROPY_COEF  = 0.01     # encourages exploration
NUM_EPISODES  = 2000
LOG_EVERY     = 50

# ── Setup ────────────────────────────────────────────────────────────────────
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


# ── Training loop ─────────────────────────────────────────────────────────────
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
```

Run it:

```bash
cd Game
python train_reinforce.py
```

Expected output after a few hundred episodes:

```
Episode   50 | loss  2.3142 | win rates ['0.25', '0.28', '0.22', '0.25']
Episode  100 | loss  1.9871 | win rates ['0.30', '0.24', '0.21', '0.25']
...
```

Win rates will fluctuate around 25% early on (pure chance). You should see one player start
pulling ahead after ~500–1000 episodes as the network learns which moves gain VP.

---

## 7. Upgrading to PPO

REINFORCE has high variance. **Proximal Policy Optimisation (PPO)** is the standard algorithm
for games like this. The simplest route is to use **Stable-Baselines3** with a custom environment
wrapper, or to implement PPO manually.

### Option A — Stable-Baselines3 (easiest)

```bash
.venv\Scripts\pip install stable-baselines3
```

SB3 expects a single-agent `gymnasium`-compatible environment. Wrap `CatanEnv` to always control player 0
and use `RandomAgent` for the other seats:

```python
# Game/sb3_wrapper.py
import gymnasium as gym
import numpy as np
from catan_env import CatanEnv, RandomAgent

class CatanSingleAgent(gym.Env):
    def __init__(self, num_players=4):
        super().__init__()
        self._env      = CatanEnv(num_players=num_players, reward_shaping=True)
        self._opponents = [RandomAgent() for _ in range(num_players - 1)]
        obs_n          = self._env.obs_size()
        act_n          = self._env.action_size()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(obs_n,), dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(act_n)

    def reset(self, *, seed=None, options=None):
        obs, mask = self._env.reset()
        self._mask = mask
        obs, mask = self._step_opponents(obs, mask)
        return obs.astype(np.float32), {"action_mask": mask}

    def step(self, action):
        rewards, done = self._env.step(int(action))
        reward = rewards.get(0, 0.0)
        obs, mask = self._env.observe() if not done else (np.zeros(self._env.obs_size(), dtype=np.float32), np.zeros(self._env.action_size(), dtype=bool))
        if not done:
            obs, mask = self._step_opponents(obs, mask)
        self._mask = mask
        return obs.astype(np.float32), reward, done, False, {"action_mask": mask}

    def _step_opponents(self, obs, mask):
        """Advance until it is player 0's turn again."""
        while not self._env.done and self._env.current_player != 0:
            o, m = self._env.observe()
            opp_slot = (self._env.current_player - 1) % (self._env.num_players - 1)
            action   = self._opponents[opp_slot].choose(o, m)
            self._env.step(action)
        if not self._env.done:
            return self._env.observe()
        return obs, mask
```

Train with PPO:

```python
# Game/train_ppo.py
from stable_baselines3 import PPO
from sb3_wrapper import CatanSingleAgent

env   = CatanSingleAgent(num_players=4)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,
    tensorboard_log="./ppo_logs/",
)
model.learn(total_timesteps=500_000)
model.save("ppo_catan")
```

### Option B — Manual PPO

Implement the PPO clipped objective in your own training loop.
The main change from REINFORCE is using a value baseline and the clipped ratio:

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ and $\hat{A}_t$ is the advantage estimate.

This prevents destructively large policy updates and is much more stable than REINFORCE for long-horizon games like Catan.

---

## 8. Tracking progress

### Tensorboard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/catan_run1")

# In training loop:
writer.add_scalar("Loss/total", loss.item(), episode)
writer.add_scalar("WinRate/player0", win_rate_0, episode)
```

```bash
tensorboard --logdir runs/
```

### Metrics to watch

| Metric                   | Good sign                                               |
| ------------------------ | ------------------------------------------------------- |
| Win rate vs. RandomAgent | Should rise above 25% within ~1000 episodes             |
| Average game length      | Should shorten as agent stops wasting moves             |
| Entropy                  | Should start high (~5.7 = log(298)) and slowly decrease |
| Value loss               | Should trend downward as value head learns              |

---

## 9. What to try next

Once you have a working PPO agent:

1. **Self-play with league** — Periodically freeze copies of the current policy and add them to the opponent pool. The agent learns to beat increasingly strong opponents.

2. **Larger network** — Try `hidden=512` or add a third hidden layer. The 950-dim input is large enough to benefit.

3. **Graph Neural Network** — The board is a fixed planar graph. A GNN message-passing over hex/vertex/edge nodes can generalise better to unseen board layouts and is the approach used in research-grade Catan bots.

4. **MCTS + value head (AlphaZero-style)** — Use `env.state()` to expand the game tree. At each node, run a few MCTS rollouts guided by the policy network's action probabilities and backed up by the value network. This is the strongest known approach for deterministic (or near-deterministic) board games.

5. **Curriculum** — Start with 2-player games (simpler) and gradually increase to 4. This can cut training time significantly.
