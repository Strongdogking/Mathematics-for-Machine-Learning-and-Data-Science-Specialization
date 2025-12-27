import numpy as np
import matplotlib.pyplot as plt

# Define the function with different bases
def f_base(x, base):
    """Calculate -x * log_base(x)"""
    if base == 'e':
        return -x * np.log(x)
    else:
        # Change of base formula: log_base(x) = ln(x) / ln(base)
        return -x * np.log(x) / np.log(base)

# Create x values (avoid x=0 as log(0) is undefined)
x = np.linspace(0.001, 3, 1000)

# Calculate y values for different bases
y_e = f_base(x, 'e')
y_2 = f_base(x, 2)
y_10 = f_base(x, 10)

# Find maximum points
# For f(x) = -x * log_base(x), the maximum ALWAYS occurs at x = 1/e
# regardless of the base, because ln(base) is a constant factor
x_max = 1 / np.e
y_max_e = f_base(x_max, 'e')
y_max_2 = f_base(x_max, 2)
y_max_10 = f_base(x_max, 10)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the three functions
plt.plot(x, y_e, 'b-', linewidth=2.5, label=r'$f(x) = -x \cdot \ln(x)$ (base e)')
plt.plot(x, y_2, 'r-', linewidth=2.5, label=r'$f(x) = -x \cdot \log_2(x)$ (base 2)')
plt.plot(x, y_10, 'g-', linewidth=2.5, label=r'$f(x) = -x \cdot \log_{10}(x)$ (base 10)')

# Add axes
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Highlight maximum points (all at x = 1/e)
plt.plot(x_max, y_max_e, 'bo', markersize=10, label=f'Max at $x = 1/e$ (base e)')
plt.plot(x_max, y_max_2, 'rs', markersize=10, label=f'Max at $x = 1/e$ (base 2)')
plt.plot(x_max, y_max_10, 'g^', markersize=10, label=f'Max at $x = 1/e$ (base 10)')

# Add grid
plt.grid(True, alpha=0.3)

# Set labels and title
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.title('Comparison of $f(x) = -x \cdot \log_{base}(x)$ with Different Bases', fontsize=16)

# Set reasonable axis limits
plt.xlim(0, 3)
plt.ylim(-1, 1)

# Add legend
plt.legend(fontsize=12, loc='upper right')

# Add text annotations
plt.text(0.15, 0.45, f'Max (base e): ({x_max:.4f}, {y_max_e:.4f})', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.text(0.55, 0.38, f'Max (base 2): ({x_max:.4f}, {y_max_2:.4f})', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
plt.text(1.5, 0.27, f'Max (base 10): ({x_max:.4f}, {y_max_10:.4f})', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Show the plot
plt.tight_layout()
plt.savefig('compare_log_bases.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("对比不同底数的对数函数 f(x) = -x * log_base(x)")
print("=" * 60)
print(f"\n以 e 为底 (自然对数):")
print(f"  最大值: {y_max_e:.4f} 出现在 x = {x_max:.4f}")
print(f"  公式: f(x) = -x * ln(x)")

print(f"\n以 2 为底:")
print(f"  最大值: {y_max_2:.4f} 出现在 x = {x_max:.4f}")
print(f"  公式: f(x) = -x * log₂(x) = -x * ln(x) / ln(2)")

print(f"\n以 10 为底:")
print(f"  最大值: {y_max_10:.4f} 出现在 x = {x_max:.4f}")
print(f"  公式: f(x) = -x * log₁₀(x) = -x * ln(x) / ln(10)")

print("\n" + "=" * 60)
print("关键观察:")
print("=" * 60)
print("1. 所有函数都在 x = 1/e 处取得最大值（与底数无关！）")
print("2. 最大值 = 1/(e * ln(base))")
print("3. 底数越大，函数值越小（因为 ln(base) 越大）")
print("4. 在 x = 1 处，所有函数值都为 0")
print("=" * 60)

print("\nPlot saved as 'compare_log_bases.png'")
