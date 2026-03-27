import os
import sys
import json
import math
import numpy as np
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


def evaluate_on_scenario(policy, device, scenario_name, n_episodes=10):
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

    return {
        "scenario": scenario_name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }


def save_model(policy, path, obs_dim, action_dim):
    clean_payload = {
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
    }
    torch.save(clean_payload, path)


def load_model(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PolicyNetwork(int(payload["obs_dim"]), int(payload["action_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
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

    training_rewards = []
    best_overall_mean = -math.inf

    for episode in range(1, num_episodes + 1):
        env = make_env()
        log_probs, rewards, entropies, total_reward, info = run_episode(
            env=env,
            policy=policy,
            device=device,
            deterministic=False,
            scenario_name=None,
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        training_rewards.append(total_reward)

        if episode % 50 == 0:
            recent_mean = float(np.mean(training_rewards[-50:]))
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Recent Mean Reward: {recent_mean:.2f} | "
                f"Last Reward: {total_reward:.2f} | "
                f"Loss: {loss.item():.4f}"
            )

        if episode % eval_freq == 0:
            scenario_results = []
            for scenario in SCENARIOS:
                result = evaluate_on_scenario(policy, device, scenario, n_episodes=5)
                scenario_results.append(result)

            overall_mean = float(np.mean([r["mean_reward"] for r in scenario_results]))
            print(f"\n[Eval @ episode {episode}] overall mean reward: {overall_mean:.2f}")
            for r in scenario_results:
                print(
                    f"  {r['scenario']}: "
                    f"mean={r['mean_reward']:.2f}, std={r['std_reward']:.2f}"
                )

            if overall_mean > best_overall_mean:
                best_overall_mean = overall_mean
                save_model(
                    policy,
                    "best_models/reinforce/best_model.pt",
                    obs_dim,
                    action_dim,
                )
                print("  Saved new best REINFORCE model.")

    save_model(policy, "models/reinforce/kitchen_reinforce_model.pt", obs_dim, action_dim)

    final_model = load_model("models/reinforce/kitchen_reinforce_model.pt", device)
    models_to_test = {
        "final_model": final_model
    }

    best_model_path = "best_models/reinforce/best_model.pt"
    if os.path.exists(best_model_path):
        models_to_test["best_model"] = load_model(best_model_path, device)

    all_results = {}
    for model_name, loaded_model in models_to_test.items():
        print(f"\n===== Evaluating {model_name} =====")
        scenario_results = []

        for scenario in SCENARIOS:
            result = evaluate_on_scenario(loaded_model, device, scenario, n_episodes=10)
            scenario_results.append(result)

            print(
                f"{scenario}: "
                f"mean={result['mean_reward']:.2f}, "
                f"std={result['std_reward']:.2f}, "
                f"min={result['min_reward']:.2f}, "
                f"max={result['max_reward']:.2f}"
            )

        overall_mean = float(np.mean([r["mean_reward"] for r in scenario_results]))
        all_results[model_name] = {
            "overall_mean_reward": overall_mean,
            "scenario_results": scenario_results,
        }
        print(f"Overall mean reward for {model_name}: {overall_mean:.2f}")

    with open("results/reinforce/reinforce_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open("results/reinforce/reinforce_training_rewards.json", "w") as f:
        json.dump(
            {
                "num_episodes": num_episodes,
                "gamma": gamma,
                "entropy_coef": entropy_coef,
                "training_rewards": training_rewards,
            },
            f,
            indent=2,
        )

    print("\nSaved evaluation results to results/reinforce/reinforce_eval_results.json")
    print("Saved training rewards to results/reinforce/reinforce_training_rewards.json")


if __name__ == "__main__":
    main()
