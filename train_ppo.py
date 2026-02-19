"""
Multi-agent PPO training loop for Catan.

Creates 4 independent PPO agents (same architecture, different random weights)
and runs them against the real CatanEnv.

Usage:
    # Start fresh:
    python train_ppo.py

    # Resume from the latest checkpoint:
    python train_ppo.py --resume

Simplified rules (handled by CatanEnv):
    - No dev-card actions
    - No robber / discard (auto-resolved internally)
    - Only 4:1 bank trades

Action space : 202 discrete actions  (see CatanEnv docstring)
Obs space    : 404-dimensional float32 vector  (see GameState.get_observation_vector)
"""

import argparse
import os
import pickle

import numpy as np

from catan_env        import CatanEnv
from agents.ppo_agent import PPOAgent


# ── Fixed dimensions (derived from CatanEnv) ─────────────────────────
OBS_DIM     = CatanEnv.OBS_DIM   # 404
ACT_DIM     = CatanEnv.ACT_DIM   # 202
NUM_PLAYERS = 4


# ── Training loop ─────────────────────────────────────────────────────

def train(
    num_episodes:   int  = 500,
    log_interval:   int  = 10,
    save_interval:  int  = 50,
    checkpoint_dir: str  = "checkpoints",
    resume:         bool = False,
) -> list:
    """
    Train NUM_PLAYERS independent PPO agents against each other.

    Checkpoints
    -----------
    Every *save_interval* episodes the agents and training state are written to
    *checkpoint_dir/*.  If *resume=True* the latest checkpoint is loaded
    automatically so training continues where it left off.

    Returns the list of trained PPOAgents.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    state_path  = os.path.join(checkpoint_dir, "training_state.pkl")
    agent_paths = [os.path.join(checkpoint_dir, f"agent_{i}.pt")
                   for i in range(NUM_PLAYERS)]

    # ── 1.  Create (or reload) environment & agents ───────────────────
    env = CatanEnv(num_players=NUM_PLAYERS)

    start_ep        = 1
    episode_rewards = [[] for _ in range(NUM_PLAYERS)]

    can_resume = resume and all(os.path.exists(p) for p in agent_paths)
    if can_resume:
        print(f"Resuming from checkpoint in '{checkpoint_dir}' …")
        agents = [PPOAgent.load_checkpoint(p) for p in agent_paths]
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                saved = pickle.load(f)
            start_ep        = saved["next_episode"]
            episode_rewards = saved["episode_rewards"]
        print(f"  Continuing from episode {start_ep}")
    else:
        agents = [PPOAgent(OBS_DIM, ACT_DIM) for _ in range(NUM_PLAYERS)]

    print(f"Device        : {agents[0].device}")
    print(f"obs_dim={OBS_DIM}  act_dim={ACT_DIM}  players={NUM_PLAYERS}")
    print(f"Episodes      : {start_ep} → {start_ep + num_episodes - 1}\n")

    # ── 2.  Episode loop ──────────────────────────────────────────────
    for ep in range(start_ep, start_ep + num_episodes):
        obs  = env.reset()
        done = False
        ep_reward = [0.0] * NUM_PLAYERS

        while not done:
            i      = env.current_player          # whose turn?
            obs_i  = obs[i]
            mask_i = env.get_action_mask(i)

            action, logp, value = agents[i].act(obs_i, mask_i)
            next_obs, rewards, done, info = env.step(action)

            reward_i = rewards[i]
            agents[i].store(obs_i, action, logp, value, reward_i, done)
            ep_reward[i] += reward_i

            obs = next_obs

        # ── 3.  End-of-episode: compute advantages & update ───────────
        for i in range(NUM_PLAYERS):
            agents[i].compute_returns_and_advantages(last_value=0.0, last_done=True)
            agents[i].update()

        for i in range(NUM_PLAYERS):
            episode_rewards[i].append(ep_reward[i])

        # ── 4.  Logging ───────────────────────────────────────────────
        if ep % log_interval == 0:
            mean_rew = [
                np.mean(episode_rewards[i][-log_interval:])
                for i in range(NUM_PLAYERS)
            ]
            rew_str = "  ".join(f"P{i}: {r:+.3f}" for i, r in enumerate(mean_rew))
            winner  = info.get("winner")
            turn    = info.get("turn", "?")
            print(
                f"Episode {ep:>5d} | {rew_str} | turns={turn}"
                + (f" | winner=P{winner}" if winner is not None else "")
            )

        # ── 5.  Checkpoint ────────────────────────────────────────────
        if ep % save_interval == 0:
            _save_checkpoint(agents, episode_rewards, ep + 1,
                             agent_paths, state_path)
            print(f"  [checkpoint saved at episode {ep}]")

    # Final save
    _save_checkpoint(agents, episode_rewards, start_ep + num_episodes,
                     agent_paths, state_path)
    print("\nTraining complete.  Final checkpoint saved.")
    return agents


# ── Checkpoint helpers ────────────────────────────────────────────────

def _save_checkpoint(agents, episode_rewards, next_episode,
                     agent_paths, state_path):
    for i, agent in enumerate(agents):
        agent.save_checkpoint(agent_paths[i])
    with open(state_path, "wb") as f:
        pickle.dump(
            {"next_episode": next_episode, "episode_rewards": episode_rewards},
            f,
        )


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agents on Catan")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to run")
    parser.add_argument("--log",      type=int, default=10,
                        help="Log every N episodes")
    parser.add_argument("--save",     type=int, default=50,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    train(
        num_episodes   = args.episodes,
        log_interval   = args.log,
        save_interval  = args.save,
        checkpoint_dir = args.ckpt_dir,
        resume         = args.resume,
    )
