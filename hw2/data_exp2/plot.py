import pandas as pd
import matplotlib.pyplot as plt
import os

csv_dir = os.path.join(os.path.dirname(__file__), "csvs")
out_dir = os.path.dirname(__file__)

steps = pd.read_csv(os.path.join(csv_dir, "cheetah_step.csv"))["Value"].values

baseline_01 = pd.read_csv(os.path.join(csv_dir, "baseline_value.csv"))["Value"].values
baseline_001 = pd.read_csv(os.path.join(csv_dir, "baseline_value_blr-0.001.csv"))["Value"].values
return_baseline = pd.read_csv(os.path.join(csv_dir, "baseline_return_value.csv"))["Value"].values
return_baseline_001 = pd.read_csv(os.path.join(csv_dir, "baseline_return_value_blr-0.001.csv"))["Value"].values
return_vanilla = pd.read_csv(os.path.join(csv_dir, "vanilla_return_value.csv"))["Value"].values

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, baseline_01, label="baseline (blr=0.01)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Baseline Value")
ax.set_title("Baseline Curve (blr=0.01)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "exp2_1_baseline_blr01.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, baseline_01, label="baseline (blr=0.01)")
ax.plot(steps, baseline_001, label="baseline (blr=0.001)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Baseline Value")
ax.set_title("Baseline Curves Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "exp2_2_baseline_compare.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, return_vanilla, label="vanilla (no baseline)")
ax.plot(steps, return_baseline, label="baseline (blr=0.01)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Eval Average Return")
ax.set_title("Return: Baseline (blr=0.01) vs Vanilla")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "exp2_3_return_vs_vanilla.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, return_baseline, label="return (blr=0.01)")
ax.plot(steps, return_baseline_001, label="return (blr=0.001)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Eval Average Return")
ax.set_title("Return Curves: Two Baseline LRs")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "exp2_4_return_two_blr.png"), dpi=150)
plt.show()
