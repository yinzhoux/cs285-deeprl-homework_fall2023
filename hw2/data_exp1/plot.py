import pandas as pd
import matplotlib.pyplot as plt
import os

configs = [
    ("cartpole", "vanilla"),
    ("cartpole_reward_to_go", "rtg"),
    ("cartpole_normalization", "na"),
    ("cartpole_normalization_reward_to_go", "rtg + na"),
]


def plot_batch(batch_dir, title, ax):
    for prefix, label in configs:
        step_file = os.path.join(batch_dir, f"{prefix}_step.csv")
        value_file = os.path.join(batch_dir, f"{prefix}.csv")
        steps = pd.read_csv(step_file)["Value"].values
        returns = pd.read_csv(value_file)["Value"].values
        ax.plot(steps, returns, label=label)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Average Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_batch("small_batch", "Small Batch (b=1000)", axes[0])
plot_batch("larg_batch", "Large Batch (b=4000)", axes[1])

plt.tight_layout()
plt.savefig("exp1_cartpole.png", dpi=150)
plt.show()
