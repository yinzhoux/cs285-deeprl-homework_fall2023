import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to CSV files
    default_csv_path = os.path.join(os.path.dirname(base_dir), "b500_lr0.01_gae0.98_mean_return.csv") # Assuming this is the hyperparam tuned one based on previous context
    # Correction: The prompt says "b500_lr0.01_gae0.98_mean_return.csv" is the ONE generated previously.
    # The attached "csvs" folder in data_exp4 contains s1_return.csv etc for the "default" configuration from an earlier experiment?
    # Actually, looking at the previous turn, the user attached "b500_lr0.01_gae0.98_mean_return.csv" directly.
    # And there is a folder "data_exp4/csvs" which likely contains the "default" run data (from the very first request). W
    # The first request was "data_exp4" -> 5 runs. I made a plot for it.
    # Now user wants to compare "default" (from data_exp4/csvs) vs "tuned" (from the new csv).
    
    # Load "Default" data (averaged from s1..s5 in data_exp4/csvs, assuming we re-calculate or load a saved csv if exists)
    # The user attached "csvs" folder again, implying I should use it.
    # I should re-use logic to load/average "default" from data_exp4/csvs.
    
    csv_dir = os.path.join(os.path.dirname(base_dir), "csvs")
    
    # Load default runs
    default_curves = []
    run_ids = ["s1", "s2", "s3", "s4", "s5"]
    for run_id in run_ids:
        step_path = os.path.join(csv_dir, f"{run_id}_step.csv")
        return_path = os.path.join(csv_dir, f"{run_id}_return.csv")
        
        if os.path.exists(step_path) and os.path.exists(return_path):
            step_df = pd.read_csv(step_path)
            return_df = pd.read_csv(return_path)
            merged = pd.merge(return_df, step_df, on="Step", suffixes=('_return', '_step'))
            # Value_return is return, Value_step is env_step
            default_curves.append((merged["Value_step"].values, merged["Value_return"].values))
            
    # Load "Tuned" data (b500_lr0.01)
    # The file path is provided in attachment as "d:\courses\cs285-deepRL\cs285-deeprl-homework_fall2023\hw2\data_exp4\hyperparam_tuning\b500_lr0.01_gae0.98_mean_return.csv"
    tuned_csv_path = os.path.join(base_dir, "b500_lr0.01_gae0.98_mean_return.csv")
    
    if not os.path.exists(tuned_csv_path):
         print(f"File not found: {tuned_csv_path}")
         return

    tuned_df = pd.read_csv(tuned_csv_path)
    
    import numpy as np
    plt.figure(figsize=(10, 6))
    
    # Plot Tuned
    p1 = plt.plot(tuned_df["env_step"], tuned_df["eval_return"], label="Tuned (b=500, lr=0.01)", linewidth=2)
    
    # Mark tuned crossing
    cross_tuned = tuned_df[tuned_df["eval_return"] >= 1000]
    if not cross_tuned.empty:
        step = cross_tuned.iloc[0]["env_step"]
        plt.axvline(x=step, color=p1[0].get_color(), linestyle='--', alpha=0.5)
        plt.text(step, 1005, f'{int(step)}', color=p1[0].get_color(), rotation=90, va='bottom', ha='right', fontsize=18)
    
    # Process and Plot Default
    if default_curves:
        # Interpolate default to same grid for averaging
        min_env = 0
        max_env = max(c[0][-1] for c in default_curves)
        common_env = np.linspace(min_env, max_env, 500)
        
        interp_returns = []
        for x, y in default_curves:
            interp_returns.append(np.interp(common_env, x, y))
            
        mean_default = np.mean(interp_returns, axis=0)
        p2 = plt.plot(common_env, mean_default, label="Default", linewidth=2)
        
        # Mark default crossing
        cross_default_indices = np.where(mean_default >= 1000)[0]
        if len(cross_default_indices) > 0:
            step = common_env[cross_default_indices[0]]
            plt.axvline(x=step, color=p2[0].get_color(), linestyle='--', alpha=0.5)
            plt.text(step, 1005, f'{int(step)}', color=p2[0].get_color(), rotation=90, va='bottom', ha='left', fontsize=18)
        
    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Average Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(base_dir, "comparison_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
