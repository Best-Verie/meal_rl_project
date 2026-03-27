import os
import sys
import json
import csv
import numpy as np

sys.path.append(os.path.abspath("."))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import KitchenMealPlanningEnv


SCENARIOS = [
    "adult_weight_loss",
    "adult_weight_gain",
    "adult_hypertension",
    "baby_meal",
]

PPO_EXPERIMENTS = [
    {"name": "ppo_exp_01", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.01, "clip_range": 0.2},
    {"name": "ppo_exp_02", "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.01, "clip_range": 0.2},
    {"name": "ppo_exp_03", "learning_rate": 5e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.01, "clip_range": 0.2},
    {"name": "ppo_exp_04", "learning_rate": 3e-4, "gamma": 0.95, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.01, "clip_range": 0.2},
    {"name": "ppo_exp_05", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64, "ent_coef": 0.01, "clip_range": 0.2},
    {"name": "ppo_exp_06", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 128, "ent_coef": 0.01, "clip_range": 0.2},
    {"name": "ppo_exp_07", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.001, "clip_range": 0.2},
    {"name": "ppo_exp_08", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.02, "clip_range": 0.2},
    {"name": "ppo_exp_09", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.01, "clip_range": 0.1},
    {"name": "ppo_exp_10", "learning_rate": 3e-4, "gamma": 0.995, "n_steps": 2048, "batch_size": 128, "ent_coef": 0.005, "clip_range": 0.25},
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
        best_model_save_path=f"best_models/ppo/{config['name']}",
        log_path=f"logs/ppo/{config['name']}",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=10,
        gamma=config["gamma"],
        gae_lambda=0.95,
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log="logs/ppo",
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    model_dir = f"models/ppo/{config['name']}"
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
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("best_models/ppo", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)
    os.makedirs("results/ppo", exist_ok=True)

    results = []

    for config in PPO_EXPERIMENTS:
        print(f"Running {config['name']} ...")
        result = run_experiment(config)
        results.append(result)
        print(result)

    json_path = "results/ppo/ppo_hyperparameter_results.json"
    csv_path = "results/ppo/ppo_hyperparameter_results.csv"

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
