
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

    os.makedirs(checkpoint_dir, exist_ok=True)
    state_path  = os.path.join(checkpoint_dir, "training_state.pkl")
    agent_paths = [os.path.join(checkpoint_dir, f"agent_{i}.pt")
                   for i in range(NUM_PLAYERS)]

    # ── 1.  Create (or reload) environment & agents ───────────────────
    env = CatanEnv(num_players=NUM_PLAYERS)

    start_ep        = 1
    episode_rewards = [[] for _ in range(NUM_PLAYERS)]

    # ── per-episode tracking buffers ─────────────────────────────────
    win_buf   = []             # winner id (-1 = timeout) per episode
    turn_buf  = []             # turns per episode
    vp_buf    = [[] for _ in range(NUM_PLAYERS)]  # final VPs per episode
    loss_buf  = {"policy": [], "value": [], "entropy": []}  # avg across agents
    # ─────────────────────────────────────────────────────────────────

    can_resume = resume and all(os.path.exists(p) for p in agent_paths)
    if can_resume:
        print(f"Resuming from checkpoint in '{checkpoint_dir}' …")
        agents = [PPOAgent.load_checkpoint(p) for p in agent_paths]
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                saved = pickle.load(f)
            start_ep        = saved["next_episode"]
            episode_rewards = saved["episode_rewards"]
            win_buf         = saved.get("win_buf",  [])
            turn_buf        = saved.get("turn_buf", [])
            vp_buf          = saved.get("vp_buf",   [[] for _ in range(NUM_PLAYERS)])
            loss_buf        = saved.get("loss_buf",
                                        {"policy": [], "value": [], "entropy": []})
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
        ep_losses = {"policy": [], "value": [], "entropy": []}
        for i in range(NUM_PLAYERS):
            agents[i].compute_returns_and_advantages(last_value=0.0, last_done=True)
            L = agents[i].update()
            ep_losses["policy"].append(L["policy_loss"])
            ep_losses["value"].append(L["value_loss"])
            ep_losses["entropy"].append(L["entropy"])

        for i in range(NUM_PLAYERS):
            episode_rewards[i].append(ep_reward[i])

        winner = info.get("winner")
        win_buf.append(winner if winner is not None else -1)
        turn_buf.append(info.get("turn", 0))
        for i in range(NUM_PLAYERS):
            vp_buf[i].append(env.game.players[i].get_total_victory_points())
        for k in loss_buf:
            loss_buf[k].append(np.mean(ep_losses[k]))

        # ── 4.  Logging ───────────────────────────────────────────────
        if ep % log_interval == 0:
            w = win_buf[-log_interval:]
            t = turn_buf[-log_interval:]
            n = len(w)

            # Win rate per player and timeout rate
            win_rates  = [w.count(i) / n for i in range(NUM_PLAYERS)]
            timeout_rt = w.count(-1) / n

            # Mean reward over window
            mean_rew = [np.mean(episode_rewards[i][-log_interval:])
                        for i in range(NUM_PLAYERS)]

            # Mean final VPs over window
            mean_vp = [np.mean(vp_buf[i][-log_interval:])
                       for i in range(NUM_PLAYERS)]

            # Mean PPO metrics over window
            mean_pi  = np.mean(loss_buf["policy"][-log_interval:])
            mean_val = np.mean(loss_buf["value"][-log_interval:])
            mean_ent = np.mean(loss_buf["entropy"][-log_interval:])
            mean_turns = np.mean(t)

            print(f"\n{'─'*70}")
            print(f" Episode {ep:>5d}  (last {n} games)")
            print(f"{'─'*70}")

            # Per-player table
            header = f"  {'':4s}  {'WinRate':>8s}  {'AvgReward':>10s}  {'AvgVP':>6s}"
            print(header)
            for i in range(NUM_PLAYERS):
                marker = " ←" if win_rates[i] == max(win_rates) else ""
                print(f"  P{i}    {win_rates[i]:>7.1%}  {mean_rew[i]:>+10.3f}  {mean_vp[i]:>6.2f}{marker}")

            print(f"  timeout          {timeout_rt:>7.1%}")
            print(f"  avg turns        {mean_turns:>7.1f}")

            # PPO health
            ent_warn = "  ⚠ low" if mean_ent < 1.0 else ""
            print(f"  policy loss      {mean_pi:>+8.4f}")
            print(f"  value  loss      {mean_val:>8.4f}")
            print(f"  entropy          {mean_ent:>8.4f}{ent_warn}")
            print(f"{'─'*70}")

        # ── 5.  Checkpoint ────────────────────────────────────────────
        if ep % save_interval == 0:
            _save_checkpoint(agents, episode_rewards, ep + 1,
                             win_buf, turn_buf, vp_buf, loss_buf,
                             agent_paths, state_path)
            print(f"  [checkpoint saved at episode {ep}]")

    # Final save
    _save_checkpoint(agents, episode_rewards, start_ep + num_episodes,
                     win_buf, turn_buf, vp_buf, loss_buf,
                     agent_paths, state_path)
    print("\nTraining complete.  Final checkpoint saved.")
    return agents


# ── Checkpoint helpers ────────────────────────────────────────────────

def _save_checkpoint(agents, episode_rewards, next_episode,
                     win_buf, turn_buf, vp_buf, loss_buf,
                     agent_paths, state_path):
    for i, agent in enumerate(agents):
        agent.save_checkpoint(agent_paths[i])
    with open(state_path, "wb") as f:
        pickle.dump(
            {
                "next_episode":    next_episode,
                "episode_rewards": episode_rewards,
                "win_buf":         win_buf,
                "turn_buf":        turn_buf,
                "vp_buf":          vp_buf,
                "loss_buf":        loss_buf,
            },
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
