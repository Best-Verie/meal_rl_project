import json
import numpy as np

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from environment.custom_env import KitchenMealPlanningEnv

sys.path.append(os.path.abspath("."))

SCENARIOS = [
    "adult_weight_loss",
    "adult_weight_gain",
    "adult_hypertension",
    "baby_meal",
]


def make_env(log_dir):
    env = KitchenMealPlanningEnv(max_steps=20)
    env = Monitor(env, log_dir)  
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
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("best_models/dqn", exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("results/dqn", exist_ok=True)

    log_dir = "logs/dqn/"

    train_env = make_env(log_dir)
    eval_env = make_env(log_dir)
    logger = logger = configure(log_dir, ["csv", "tensorboard"])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="best_models/dqn",
        log_path="logs/dqn/eval",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=5e-4,
        buffer_size=20000,
        learning_starts=500,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="logs/dqn",
        device="auto",
    )

    model.set_logger(logger)
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("models/dqn/kitchen_dqn_model")

    final_model = DQN.load("models/dqn/kitchen_dqn_model")
    best_model_path = "best_models/dqn/best_model.zip"

    models_to_test = {
        "final_model": final_model
    }

    if os.path.exists(best_model_path):
        models_to_test["best_model"] = DQN.load(best_model_path)

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

    with open("results/dqn/dqn_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved evaluation results to results/dqn/dqn_eval_results.json")


if __name__ == "__main__":
    main()
