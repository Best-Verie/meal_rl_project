import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.custom_env import KitchenMealPlanningEnv


SCENARIOS = [
    "adult_weight_loss",
    "adult_weight_gain",
    "adult_hypertension",
    "baby_meal",
]


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


def make_env():
    return KitchenMealPlanningEnv(max_steps=20)


def run_episode(env, policy, device, deterministic=False, scenario_name=None):
    if scenario_name is not None:
        obs, info = env.reset(options={"scenario_name": scenario_name})
    else:
        obs, info = env.reset()

    log_probs = []
    rewards = []
    entropies = []

    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = policy(obs_tensor)

        if deterministic:
            action = torch.argmax(dist.probs, dim=1).item()
            log_prob = torch.log(dist.probs[0, action] + 1e-8)
            entropy = dist.entropy().mean()
        else:
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor).squeeze()
            entropy = dist.entropy().mean()

        obs, reward, done, truncated, info = env.step(int(action))

        log_probs.append(log_prob)
        rewards.append(float(reward))
        entropies.append(entropy)
        total_reward += reward

    return log_probs, rewards, entropies, total_reward, info


def compute_returns(rewards, gamma, device):
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    if len(returns) > 1 and torch.std(returns) > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def evaluate_on_scenario(policy, device, scenario_name, n_episodes=5):
    rewards = []

    for _ in range(n_episodes):
        env = make_env()
        _, _, _, total_reward, _ = run_episode(
            env=env,
            policy=policy,
            device=device,
            deterministic=True,
            scenario_name=scenario_name,
        )
        rewards.append(total_reward)

    return float(np.mean(rewards))

def save_model(policy, path, obs_dim, action_dim):
    clean_payload = {
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
    }
    torch.save(clean_payload, path)


def load_model(path, device):
    payload = torch.load(path, map_location=device)

    # CASE 1: New format (correct)
    if isinstance(payload, dict) and "state_dict" in payload:
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        state_dict = payload["state_dict"]

    # CASE 2: Old format (weights only)
    else:
        env = KitchenMealPlanningEnv()
        obs, _ = env.reset()

        obs_dim = int(obs.shape[0])
        action_dim = int(env.action_space.n)
        state_dict = payload

    model = PolicyNetwork(obs_dim, action_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
    
def main():
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("best_models/reinforce", exist_ok=True)
    os.makedirs("logs/reinforce", exist_ok=True)
    os.makedirs("results/reinforce", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp_env = make_env()
    obs, _ = temp_env.reset()
    obs_dim = int(obs.shape[0])
    action_dim = int(temp_env.action_space.n)

    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    num_episodes = 3000
    gamma = 0.99
    entropy_coef = 0.001
    eval_freq = 200

    #  LOG STORAGE
    reward_log = []
    loss_log = []
    entropy_log = []
    eval_log = []

    best_score = -math.inf

    for episode in range(1, num_episodes + 1):
        env = make_env()

        log_probs, rewards, entropies, total_reward, _ = run_episode(
            env, policy, device
        )

        returns = compute_returns(rewards, gamma, device)

        policy_loss_terms = []
        entropy_terms = []

        for log_prob, G, entropy in zip(log_probs, returns, entropies):
            policy_loss_terms.append(-log_prob * G)
            entropy_terms.append(entropy)

        policy_loss = torch.stack(policy_loss_terms).sum()
        entropy_bonus = torch.stack(entropy_terms).mean()

        loss = policy_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        #  SAVE LOGS
        reward_log.append(total_reward)
        loss_log.append(loss.item())
        entropy_log.append(entropy_bonus.item())

        # PRINT
        if episode % 50 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f}")

        #  EVALUATION LOG (VERY IMPORTANT)
        if episode % eval_freq == 0:
            scores = []

            for scenario in SCENARIOS:
                score = evaluate_on_scenario(policy, device, scenario)
                scores.append(score)

            mean_score = float(np.mean(scores))

            eval_log.append({
                "episode": episode,
                "mean_reward": mean_score
            })

            print(f"[EVAL] Episode {episode} → Mean Reward: {mean_score:.2f}")

            if mean_score > best_score:
                best_score = mean_score
                torch.save(policy.state_dict(), "best_models/reinforce/best_model.pt")

    #  SAVE FINAL MODEL
    save_model(policy, "best_models/reinforce/best_model.pt", obs_dim, action_dim)  
    # SAVE CSV FILES (THIS IS WHAT YOU WERE MISSING)

    pd.DataFrame({
        "episode": list(range(1, len(reward_log) + 1)),
        "reward": reward_log
    }).to_csv("logs/reinforce/monitor.csv", index=False)

    pd.DataFrame({
        "episode": list(range(1, len(loss_log) + 1)),
        "loss": loss_log,
        "entropy": entropy_log
    }).to_csv("logs/reinforce/progress.csv", index=False)

    pd.DataFrame(eval_log).to_csv("logs/reinforce/eval.csv", index=False)

    print("Logs saved (monitor.csv, progress.csv, eval.csv)")


if __name__ == "__main__":
    main()
