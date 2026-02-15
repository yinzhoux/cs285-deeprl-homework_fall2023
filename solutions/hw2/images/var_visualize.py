import numpy as np
import matplotlib.pyplot as plt

def plot_functions():
    # Define theta range (0 to 1), avoiding endpoints
    theta = np.linspace(0.001, 0.999, 1000)

    # Function 1: Vanilla Policy Gradient Variance
    # (4*theta^2 + 8*theta + 1) / (theta * (1 - theta)^4)
    var_vanilla = (4 * theta**2 + 8 * theta + 1) / (theta * (1 - theta)**4)

    # Function 2: Reward-to-go Variance
    # (theta^2 + 3*theta + 1) / (theta * (1 - theta)^4)
    var_rtg = (theta**2 + 3 * theta + 1) / (theta * (1 - theta)**4)

    plt.figure(figsize=(10, 6))
    
    # Plot the functions
    plt.plot(theta, var_vanilla, label=r'Vanilla PG Variance: $\frac{4\theta^2 + 8\theta + 1}{\theta(1-\theta)^4}$', linewidth=2)
    plt.plot(theta, var_rtg, label=r'Reward-to-go Variance: $\frac{\theta^2 + 3\theta + 1}{\theta(1-\theta)^4}$', linewidth=2, linestyle='--')

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    plt.xlabel(r'$\theta$ (0 to 1)')
    plt.ylabel('Variance (log scale)')
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_functions()
