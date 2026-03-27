import pandas as pd
import matplotlib.pyplot as plt

dqn = pd.read_csv("results/dqn/dqn_hyperparameter_results.csv")
ppo = pd.read_csv("results/ppo/ppo_hyperparameter_results.csv")
reinforce = pd.read_csv("results/reinforce/reinforce_hyperparameter_results.csv")

# Overall comparison
plt.figure()
plt.bar(
    ["DQN", "PPO", "REINFORCE"],
    [
        dqn["overall_mean_reward"].max(),
        ppo["overall_mean_reward"].max(),
        reinforce["overall_mean_reward"].max(),
    ]
)
plt.title("Best Algorithm Comparison")
plt.ylabel("Mean Reward")
plt.savefig("results/overall_comparison.png")
plt.show()

# Example: learning rate effect (DQN)
plt.figure()
plt.scatter(dqn["learning_rate"], dqn["overall_mean_reward"])
plt.title("DQN Learning Rate vs Reward")
plt.xlabel("Learning Rate")
plt.ylabel("Reward")
plt.savefig("results/dqn_lr_plot.png")
plt.show()
