import torch
import torch.nn as nn
import torch.nn.functional as F

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

