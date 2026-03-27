import os
import sys
import json
import csv
import numpy as np

sys.path.append(os.path.abspath("."))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import KitchenMealPlanningEnv


SCENARIOS = [
    "adult_weight_loss",
    "adult_weight_gain",
    "adult_hypertension",
    "baby_meal",
]

DQN_EXPERIMENTS = [
    {"name": "dqn_exp_01", "learning_rate": 1e-3, "gamma": 0.95, "buffer_size": 10000, "batch_size": 32, "exploration_fraction": 0.30, "target_update_interval": 500},
    {"name": "dqn_exp_02", "learning_rate": 5e-4, "gamma": 0.95, "buffer_size": 10000, "batch_size": 64, "exploration_fraction": 0.30, "target_update_interval": 500},
    {"name": "dqn_exp_03", "learning_rate": 1e-4, "gamma": 0.95, "buffer_size": 10000, "batch_size": 64, "exploration_fraction": 0.30, "target_update_interval": 500},
    {"name": "dqn_exp_04", "learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 10000, "batch_size": 64, "exploration_fraction": 0.30, "target_update_interval": 500},
    {"name": "dqn_exp_05", "learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 20000, "batch_size": 64, "exploration_fraction": 0.40, "target_update_interval": 500},
    {"name": "dqn_exp_06", "learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 20000, "batch_size": 128, "exploration_fraction": 0.40, "target_update_interval": 500},
    {"name": "dqn_exp_07", "learning_rate": 3e-4, "gamma": 0.99, "buffer_size": 30000, "batch_size": 64, "exploration_fraction": 0.40, "target_update_interval": 250},
    {"name": "dqn_exp_08", "learning_rate": 3e-4, "gamma": 0.99, "buffer_size": 30000, "batch_size": 64, "exploration_fraction": 0.20, "target_update_interval": 250},
    {"name": "dqn_exp_09", "learning_rate": 1e-4, "gamma": 0.995, "buffer_size": 30000, "batch_size": 128, "exploration_fraction": 0.20, "target_update_interval": 250},
    {"name": "dqn_exp_10", "learning_rate": 7e-4, "gamma": 0.98, "buffer_size": 15000, "batch_size": 64, "exploration_fraction": 0.35, "target_update_interval": 750},
]


def make_env():
    env = KitchenMealPlanningEnv(max_steps=20)
    env = Monitor(env)
    return env


def evaluate_on_scenario(model, scenario_name, n_episodes=5):
    rewards = []

    for _ in range(n_episodes):
        env = KitchenMealPlanningEnv(max_steps=20)
        obs, _ = env.reset(options={"scenario_name": scenario_name})

        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            total_reward += reward

        rewards.append(total_reward)

    return float(np.mean(rewards))


def run_experiment(config, total_timesteps=30000):
    train_env = make_env()
    eval_env = make_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"best_models/dqn/{config['name']}",
        log_path=f"logs/dqn/{config['name']}",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=500,
        batch_size=config["batch_size"],
        tau=1.0,
        gamma=config["gamma"],
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        tensorboard_log="logs/dqn",
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    model_dir = f"models/dqn/{config['name']}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/model")

    scenario_scores = {}
    for scenario in SCENARIOS:
        scenario_scores[scenario] = evaluate_on_scenario(model, scenario, n_episodes=5)

    overall_mean = float(np.mean(list(scenario_scores.values())))

    result = {
        **config,
        "overall_mean_reward": overall_mean,
        **scenario_scores,
    }
    return result


def main():
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("best_models/dqn", exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("results/dqn", exist_ok=True)

    results = []

    for config in DQN_EXPERIMENTS:
        print(f"Running {config['name']} ...")
        result = run_experiment(config)
        results.append(result)
        print(result)

    json_path = "results/dqn/dqn_hyperparameter_results.json"
    csv_path = "results/dqn/dqn_hyperparameter_results.csv"

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
