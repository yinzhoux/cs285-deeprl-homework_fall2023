import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def _normalize_tag(tag):
    return "".join(ch.lower() for ch in tag if ch.isalnum())


def _find_tag(tags, candidates):
    norm_map = {_normalize_tag(tag): tag for tag in tags}
    for cand in candidates:
        norm = _normalize_tag(cand)
        if norm in norm_map:
            return norm_map[norm]
    return None


def load_run_data(run_dir):
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"Warning: No event files found in {run_dir}")
        return None

    try:
        ea = event_accumulator.EventAccumulator(event_files[0], size_guidance={"scalars": 0})
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])

        eval_tag = _find_tag(
            scalar_tags,
            [
                "Eval_AverageReturn",
                "eval_average_return",
                "evalreturn",
                "eval/return",
                "eval_return",
            ],
        )
        env_tag = _find_tag(
            scalar_tags,
            [
                "Train_EnvstepsSoFar",
                "train_envstepssofar",
                "env_step",
                "envsteps",
                "total_envsteps",
            ],
        )

        if eval_tag is None or env_tag is None:
            # print(f"Warning: Missing tags in {run_dir}. Found: {scalar_tags}")
            return None

        eval_events = ea.Scalars(eval_tag)
        env_events = ea.Scalars(env_tag)

        # Create a mapping from step (iteration) to environment step
        env_by_step = {e.step: e.value for e in env_events}
        
        env_steps = []
        eval_values = []
        
        for e in eval_events:
            if e.step in env_by_step:
                env_steps.append(env_by_step[e.step])
                eval_values.append(e.value)

        if not env_steps:
            return None
            
        # Sort by env steps just in case
        sorted_indices = np.argsort(env_steps)
        return np.array(env_steps)[sorted_indices], np.array(eval_values)[sorted_indices]

    except Exception as e:
        print(f"Error reading {run_dir}: {e}")
        return None


def parse_run_config(dirname):
    # Expected format: q2_pg_speed_test_b1000_lr0.01_gae0.98_s1_InvertedPendulum...
    # We want to group by b, lr, gae
    
    # Regex to extract parameters
    # Looks for _b<number>_, _lr<number>_, _gae<number>_
    b_match = re.search(r"_b(\d+)_", dirname)
    lr_match = re.search(r"_lr([\d\.]+)_", dirname)
    gae_match = re.search(r"_gae([\d\.]+)_", dirname)
    
    if b_match and lr_match and gae_match:
        b = int(b_match.group(1))
        lr = float(lr_match.group(1))
        gae = float(gae_match.group(1))
        return (b, lr, gae)
    return None


def main():
    # Adjust this path to where your data folder is
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data") 
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # Find all run directories
    run_dirs = [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
    
    # Group runs by configuration
    experiments = {}
    
    for run_dir in run_dirs:
        dirname = os.path.basename(run_dir)
        config = parse_run_config(dirname)
        
        if config:
            if config not in experiments:
                experiments[config] = []
            
            data = load_run_data(run_dir)
            if data is not None:
                experiments[config].append(data)
    
    if not experiments:
        print("No valid experiments found.")
        return

    # Group by batch size
    batches = {}
    for config, runs_data in experiments.items():
        batch_size, lr, gae = config
        if batch_size not in batches:
            batches[batch_size] = []
        batches[batch_size].append((config, runs_data))

    # Plotting per batch size
    for batch_size, batch_configs in sorted(batches.items()):
        plt.figure(figsize=(10, 6))
        
        # Sort by LR
        batch_configs.sort(key=lambda x: x[0][1])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(batch_configs)))
        
        for idx, (config, runs_data) in enumerate(batch_configs):
            _, lr, gae = config
            
            if len(runs_data) == 0:
                continue
                
            print(f"Processing config: Batch={batch_size}, LR={lr}, GAE={gae} ({len(runs_data)} runs)")

            # Interpolate to common grid
            min_env = 0
            max_env = 0
            for x, y in runs_data:
                if len(x) > 0:
                    max_env = max(max_env, x[-1])
            
            if max_env == 0:
                continue

            # Define a common grid
            common_env = np.linspace(min_env, max_env, 500)
            
            interp_returns = []
            for x, y in runs_data:
                if len(x) > 1: # Need at least 2 points to interpolate
                    interp_returns.append(np.interp(common_env, x, y))
            
            if not interp_returns:
                continue
                
            mean_return = np.mean(interp_returns, axis=0)
            std_return = np.std(interp_returns, axis=0)
            
            # Save CSV for specific config (Batch=500, LR=0.01)
            if batch_size == 500 and abs(lr - 0.01) < 1e-6:
                csv_filename = f"b{batch_size}_lr{lr}_gae{gae}_mean_return.csv"
                csv_path = os.path.join(current_dir, csv_filename)
                df_out = pd.DataFrame({"env_step": common_env, "eval_return": mean_return})
                df_out.to_csv(csv_path, index=False)
                print(f"Saved specific CSV to {csv_path}")

            label = f"lr={lr}, gae={gae}"
            plt.plot(common_env, mean_return, label=label, color=colors[idx])
            plt.fill_between(common_env, mean_return - std_return, mean_return + std_return, color=colors[idx], alpha=0.1)

            # Mark first time reaching 1000
            # We use a threshold slightly less than 1000 to catch it if it hovers near 1000 or if 1000 is the max
            # But user said "reach 1000". Since it's mean, it might be slightly less if variance is high.
            # Let's use strict >= 1000.
            crossing_indices = np.where(mean_return >= 1000)[0]
            if len(crossing_indices) > 0:
                first_idx = crossing_indices[0]
                crossing_step = common_env[first_idx]
                plt.axvline(x=crossing_step, color=colors[idx], linestyle='--', alpha=0.5)
                # Text position: step, and slightly above 1000 or at 1000
                plt.text(crossing_step, 1005, f'{int(crossing_step)}', color=colors[idx], 
                         rotation=90, va='bottom', ha='right', fontsize=8)

        plt.xlabel("Environment Steps")
        plt.ylabel("Eval Average Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(current_dir, f"hyperparam_tuning_b{batch_size}.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        plt.show()  # Try to show if environment supports it, but saving is main goal
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    main()
