import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return -x * np.log(x)

# Create x values (avoid x=0 as log(0) is undefined)
x = np.linspace(0.001, 5, 1000)
y = f(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='$f(x) = -x \cdot \log(x)$')

# Add axes
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Highlight maximum point
x_max = 1 / np.e
y_max = f(x_max)
plt.plot(x_max, y_max, 'ro', markersize=8, label=f'Maximum at $x = 1/e$')

# Add grid
plt.grid(True, alpha=0.3)

# Set labels and title
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Plot of $f(x) = -x \cdot \log(x)$', fontsize=14)

# Set reasonable axis limits
plt.xlim(0, 5)
plt.ylim(-2, 1)

# Add legend
plt.legend(fontsize=10)

# Show the plot
plt.tight_layout()
plt.savefig('negative_x_log_x.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Maximum value: {y_max:.4f} at x = {x_max:.4f}")
print("Plot saved as 'negative_x_log_x.png'")
