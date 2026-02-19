

from typing import Optional, Tuple, List
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()

        self.torso = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.torso:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.torso(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
        batch_size: int = 2048,
        max_grad_norm: float = 0.5,
    ) -> None:
        # Store hyper-parameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self._clear_buffers()

    def _clear_buffers(self) -> None:
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.val_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.done_buf: List[bool] = []
        self.adv_buf: Optional[np.ndarray] = None
        self.ret_buf: Optional[np.ndarray] = None

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)          # (1, act_dim), (1, 1)
        logits = logits.squeeze(0)               # (act_dim,)

        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            logits[~mask_t] = -1e9

        dist = Categorical(logits=logits)
        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action).item(),
            value.squeeze(-1).item(),
        )

    def store(
        self,
        obs: np.ndarray,
        action: int,
        logp: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.logp_buf.append(logp)
        self.val_buf.append(value)
        self.rew_buf.append(reward)
        self.done_buf.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        last_done: bool = True,
    ) -> None:
        n = len(self.rew_buf)
        if n == 0:
            self.adv_buf = np.array([], dtype=np.float32)
            self.ret_buf = np.array([], dtype=np.float32)
            return

        rewards = np.array(self.rew_buf, dtype=np.float32)
        values = np.array(self.val_buf, dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = float(last_done)
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        self.adv_buf = advantages
        self.ret_buf = advantages + values

    def update(self) -> dict:
        n = len(self.obs_buf)
        if n == 0:
            self._clear_buffers()
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "total_loss": 0.0}

        # Convert buffers to tensors
        obs_t = torch.as_tensor(
            np.array(self.obs_buf), dtype=torch.float32, device=self.device
        )
        act_t = torch.as_tensor(
            np.array(self.act_buf), dtype=torch.long, device=self.device
        )
        old_logp_t = torch.as_tensor(
            np.array(self.logp_buf), dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(
            self.adv_buf, dtype=torch.float32, device=self.device
        )
        ret_t = torch.as_tensor(
            self.ret_buf, dtype=torch.float32, device=self.device
        )

        # Normalise advantages — reduces variance across the minibatch and
        # makes the clipping threshold more meaningful.
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Logging accumulators
        sum_pi_loss = 0.0
        sum_v_loss = 0.0
        sum_entropy = 0.0
        sum_total = 0.0
        num_updates = 0

        for _ in range(self.epochs):
            # Shuffle indices for each epoch
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                # Forward pass — recompute logits & values for the minibatch.
                logits, values = self.net(mb_obs)
                values = values.squeeze(-1)        # (mb,)

                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)   # (mb,)

                ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, mb_ret)

                entropy = dist.entropy().mean()

                # ---- Total loss ----
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Backprop + gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                # Accumulate for logging
                sum_pi_loss += policy_loss.item()
                sum_v_loss += value_loss.item()
                sum_entropy += entropy.item()
                sum_total += loss.item()
                num_updates += 1

        # Clear buffers after update — data is consumed.
        self._clear_buffers()

        denom = max(num_updates, 1)
        return {
            "policy_loss": sum_pi_loss / denom,
            "value_loss": sum_v_loss / denom,
            "entropy": sum_entropy / denom,
            "total_loss": sum_total / denom,
        }

    # ── Checkpoint helpers ────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Save model weights, optimizer state, and hyper-parameters to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "net_state_dict":       self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # store enough hyper-params to reconstruct the agent
                "hparams": {
                    "obs_dim":       self.net.torso[0].in_features,
                    "act_dim":       self.net.policy_head.out_features,
                    "gamma":         self.gamma,
                    "lam":           self.lam,
                    "clip_eps":      self.clip_eps,
                    "value_coef":    self.value_coef,
                    "entropy_coef":  self.entropy_coef,
                    "epochs":        self.epochs,
                    "batch_size":    self.batch_size,
                    "max_grad_norm": self.max_grad_norm,
                },
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, lr: float = 3e-4) -> "PPOAgent":
        """
        Reconstruct a PPOAgent from a checkpoint saved by *save_checkpoint*.
        Optimizer learning-rate can be overridden via *lr*.
        """
        data   = torch.load(path, map_location="cpu", weights_only=False)
        hp     = data["hparams"]
        agent  = cls(
            obs_dim        = hp["obs_dim"],
            act_dim        = hp["act_dim"],
            gamma          = hp["gamma"],
            lam            = hp["lam"],
            lr             = lr,
            clip_eps       = hp["clip_eps"],
            value_coef     = hp["value_coef"],
            entropy_coef   = hp["entropy_coef"],
            epochs         = hp["epochs"],
            batch_size     = hp["batch_size"],
            max_grad_norm  = hp["max_grad_norm"],
        )
        agent.net.load_state_dict(data["net_state_dict"])
        agent.optimizer.load_state_dict(data["optimizer_state_dict"])
        # Move optimizer state to the same device as the model
        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(agent.device)
        return agent
