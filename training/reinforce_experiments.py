import os
import sys
import json
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath("."))

from environment.custom_env import KitchenMealPlanningEnv


SCENARIOS = [
    "adult_weight_loss",
    "adult_weight_gain",
    "adult_hypertension",
    "baby_meal",
]

REINFORCE_EXPERIMENTS = [
    {"name": "reinforce_exp_01", "learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.001},
    {"name": "reinforce_exp_02", "learning_rate": 5e-4, "gamma": 0.99, "entropy_coef": 0.001},
    {"name": "reinforce_exp_03", "learning_rate": 1e-4, "gamma": 0.99, "entropy_coef": 0.001},
    {"name": "reinforce_exp_04", "learning_rate": 1e-3, "gamma": 0.95, "entropy_coef": 0.001},
    {"name": "reinforce_exp_05", "learning_rate": 1e-3, "gamma": 0.995, "entropy_coef": 0.001},
    {"name": "reinforce_exp_06", "learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.0},
    {"name": "reinforce_exp_07", "learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.005},
    {"name": "reinforce_exp_08", "learning_rate": 5e-4, "gamma": 0.95, "entropy_coef": 0.005},
    {"name": "reinforce_exp_09", "learning_rate": 5e-4, "gamma": 0.995, "entropy_coef": 0.0005},
    {"name": "reinforce_exp_10", "learning_rate": 1e-4, "gamma": 0.995, "entropy_coef": 0.005},
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

    return log_probs, rewards, entropies, total_reward


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
        _, _, _, total_reward = run_episode(
            env=env,
            policy=policy,
            device=device,
            deterministic=True,
            scenario_name=scenario_name,
        )
        rewards.append(total_reward)

    return float(np.mean(rewards))


def run_experiment(config, num_episodes=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp_env = make_env()
    obs, _ = temp_env.reset()
    obs_dim = obs.shape[0]
    action_dim = temp_env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    gamma = config["gamma"]
    entropy_coef = config["entropy_coef"]

    for episode in range(1, num_episodes + 1):
        env = make_env()
        log_probs, rewards, entropies, total_reward = run_episode(env, policy, device)

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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    scenario_scores = {}
    for scenario in SCENARIOS:
        scenario_scores[scenario] = evaluate_on_scenario(policy, device, scenario, n_episodes=5)

    overall_mean = float(np.mean(list(scenario_scores.values())))

    model_dir = f"models/reinforce/{config['name']}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }, f"{model_dir}/model.pt")

    result = {
        **config,
        "overall_mean_reward": overall_mean,
        **scenario_scores,
    }
    return result


def main():
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("results/reinforce", exist_ok=True)

    results = []

    for config in REINFORCE_EXPERIMENTS:
        print(f"Running {config['name']} ...")
        result = run_experiment(config)
        results.append(result)
        print(result)

    json_path = "results/reinforce/reinforce_hyperparameter_results.json"
    csv_path = "results/reinforce/reinforce_hyperparameter_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved JSON to {json_path}")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
