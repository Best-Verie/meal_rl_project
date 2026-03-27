import os
import sys
import torch

sys.path.append(os.path.abspath("."))

from stable_baselines3 import DQN, PPO
from environment.custom_env import KitchenMealPlanningEnv
from training.reinforce_training import load_model as load_reinforce


def load_dqn():
    env = KitchenMealPlanningEnv()
    model = DQN.load("best_models/dqn/best_model.zip")
    return env, model


def load_ppo():
    env = KitchenMealPlanningEnv()
    model = PPO.load("best_models/ppo/best_model.zip")
    return env, model


def load_reinforce_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KitchenMealPlanningEnv()
    model = load_reinforce("best_models/reinforce/best_model.pt", device)
    return env, model, device


def main():
    os.makedirs("results/demo_frames", exist_ok=True)

    algorithm = "ppo"  # choose: "dqn", "ppo", "reinforce"
    scenario = "adult_hypertension"

    if algorithm == "dqn":
        env, model = load_dqn()
        device = None
    elif algorithm == "ppo":
        env, model = load_ppo()
        device = None
    elif algorithm == "reinforce":
        env, model, device = load_reinforce_model()
    else:
        raise ValueError("Invalid algorithm. Choose from dqn, ppo, reinforce.")

    obs, info = env.reset(options={"scenario_name": scenario})
    done = False
    truncated = False
    total_reward = 0.0
    step = 0

    print(f"\nRunning {algorithm.upper()} on {scenario}\n")

    while not (done or truncated):
        env.render(save_path=f"results/demo_frames/frame_{step:03d}.png")

        if algorithm in ["dqn", "ppo"]:
            action, _ = model.predict(obs, deterministic=True)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = model(obs_tensor)
            action = torch.argmax(dist.probs, dim=1).item()

        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward

        print(f"Step {step} | Reward: {reward:.4f}")
        print(f"Nutrition: {info['nutrition']}")
        print(f"Quantities: {info['quantities']}")
        print("-" * 60)

        step += 1

    env.render(save_path=f"results/demo_frames/frame_{step:03d}.png")

    print("\nFinal Total Reward:", total_reward)
    print("Frames saved in results/demo_frames/")


if __name__ == "__main__":
    main()
