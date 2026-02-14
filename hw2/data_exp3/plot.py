import pandas as pd
import matplotlib.pyplot as plt
import os

csv_dir = os.path.join(os.path.dirname(__file__), "csvs")
out_dir = os.path.dirname(__file__)

lambdas = [0, 0.95, 0.98, 0.99, 1]

fig, axes = plt.subplots(2, 5, figsize=(28, 8))

# Row 0: Baseline Loss
for i, lam in enumerate(lambdas):
    ax = axes[0, i]
    step_name = f"lambda{lam}_step.csv" if lam != 0 else "lambda0_step.csv"
    loss_name = f"lambda-{lam}_baseline-loss.csv"
    steps = pd.read_csv(os.path.join(csv_dir, step_name))["Value"].values
    loss = pd.read_csv(os.path.join(csv_dir, loss_name))["Value"].values
    ax.plot(steps, loss)
    ax.set_title(f"Baseline Loss (λ={lam})")
    ax.set_xlabel("Steps")
    if i == 0:
        ax.set_ylabel("Baseline Loss")
    ax.grid(True, alpha=0.3)

# Row 1: Return
for i, lam in enumerate(lambdas):
    ax = axes[1, i]
    step_name = f"lambda{lam}_step.csv" if lam != 0 else "lambda0_step.csv"
    ret_name = f"lambda-{lam}_return.csv"
    steps = pd.read_csv(os.path.join(csv_dir, step_name))["Value"].values
    returns = pd.read_csv(os.path.join(csv_dir, ret_name))["Value"].values
    ax.plot(steps, returns)
    ax.set_title(f"Return (λ={lam})")
    ax.set_xlabel("Steps")
    if i == 0:
        ax.set_ylabel("Eval Average Return")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "exp3_all.png"), dpi=150)
plt.show()
