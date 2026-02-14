import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_curve(csv_dir, run_id):
	step_path = os.path.join(csv_dir, f"{run_id}_step.csv")
	return_path = os.path.join(csv_dir, f"{run_id}_return.csv")

	step_df = pd.read_csv(step_path)
	return_df = pd.read_csv(return_path)

	merged = pd.merge(
		return_df[["Step", "Value"]].rename(columns={"Value": "eval_return"}),
		step_df[["Step", "Value"]].rename(columns={"Value": "env_step"}),
		on="Step",
		how="inner",
	)
	merged = merged.sort_values("env_step")
	return merged["env_step"].values, merged["eval_return"].values


def main():
	base_dir = os.path.dirname(__file__)
	csv_dir = os.path.join(base_dir, "csvs")

	step_files = glob.glob(os.path.join(csv_dir, "s*_step.csv"))
	run_ids = sorted(os.path.basename(p).replace("_step.csv", "") for p in step_files)
	if not run_ids:
		raise FileNotFoundError("No CSV files found in data_exp4/csvs")

	curves = [load_curve(csv_dir, run_id) for run_id in run_ids]

	min_env = max(x[0] for x, _ in curves)
	max_env = min(x[-1] for x, _ in curves)
	if min_env >= max_env:
		raise ValueError("Env step ranges do not overlap across runs")

	grid_size = min(len(x) for x, _ in curves)
	common_env = np.linspace(min_env, max_env, grid_size)

	interp_returns = [np.interp(common_env, x, y) for x, y in curves]
	mean_return = np.mean(interp_returns, axis=0)

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(common_env, mean_return, label=f"mean ({len(run_ids)} runs)")
	ax.set_xlabel("Environment Steps")
	ax.set_ylabel("Eval Average Return")
	ax.set_title("Eval Return vs Env Steps (Average)")
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	out_path = os.path.join(base_dir, "avg_evalreturn_envstep.png")
	plt.savefig(out_path, dpi=150)
	plt.show()


if __name__ == "__main__":
	main()
