import os
from environment.custom_env import KitchenMealPlanningEnv

os.makedirs("results/random_frames", exist_ok=True)

env = KitchenMealPlanningEnv(max_steps=12)
obs, info = env.reset(options={"scenario_name": "adult_weight_loss"})

done = False
truncated = False
step = 0
total_reward = 0.0

while not (done or truncated):
    env.render(save_path=f"results/random_frames/frame_{step:03d}.png")

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    total_reward += reward
    step += 1

env.render(save_path=f"results/random_frames/frame_{step:03d}.png")

print("Random demo complete.")
print("Frames saved in results/random_frames/")
print("Total reward:", total_reward)
