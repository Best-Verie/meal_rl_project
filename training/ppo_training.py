import os
import json
import numpy as np

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
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


def make_env():
    env = KitchenMealPlanningEnv(max_steps=20)
    env = Monitor(env)
    return env


def evaluate_on_scenario(model, scenario_name, n_episodes=10):
    rewards = []

    for _ in range(n_episodes):
        env = KitchenMealPlanningEnv(max_steps=20)
        obs, info = env.reset(options={"scenario_name": scenario_name})

        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward

        rewards.append(total_reward)

    return {
        "scenario": scenario_name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }


def main():
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("best_models/ppo", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)
    os.makedirs("results/ppo", exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="best_models/ppo",
        log_path="logs/ppo/eval",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/ppo",
        device="auto",
    )

    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("models/ppo/kitchen_ppo_model")

    final_model = PPO.load("models/ppo/kitchen_ppo_model")
    best_model_path = "best_models/ppo/best_model.zip"

    models_to_test = {
        "final_model": final_model
    }

    if os.path.exists(best_model_path):
        models_to_test["best_model"] = PPO.load(best_model_path)

    all_results = {}

    for model_name, loaded_model in models_to_test.items():
        print(f"\n===== Evaluating {model_name} =====")
        scenario_results = []

        for scenario in SCENARIOS:
            result = evaluate_on_scenario(loaded_model, scenario, n_episodes=10)
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

    with open("results/ppo/ppo_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved evaluation results to results/ppo/ppo_eval_results.json")


if __name__ == "__main__":
    main()
