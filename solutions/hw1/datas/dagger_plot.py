import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data(filename, value_col):
    steps = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['Step']))
            values.append(float(row[value_col]))
    return np.array(steps), np.array(values)

def plot_trend():
    try:
        steps, mean_rets = load_data('mean_ret_ant.csv', 'Mean Return')
        _, std_rets = load_data('std_ret_ant.csv', 'Std Return')
        
        plt.figure(figsize=(10, 6))
        
        # Plot mean line
        plt.plot(steps, mean_rets, label='Mean Return', color='tab:blue', linewidth=2)
        
        # Plot shaded area for standard deviation
        plt.fill_between(steps, mean_rets - std_rets, mean_rets + std_rets, 
                         alpha=0.2, color='tab:blue', label='Standard Deviation')
        
        # Plot baseline
        plt.axhline(y=4682, color='tab:red', linestyle='--', linewidth=2, label=f'Baseline ({4682})')
        
        plt.xlabel('Step / Iteration', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_file = '../images/ant_dagger.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_trend()
