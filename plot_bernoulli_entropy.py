import numpy as np
import matplotlib.pyplot as plt

def bernoulli_entropy(p, base='e'):
    """
    Calculate Bernoulli entropy H(p) = -p * log_base(p) - (1-p) * log_base(1-p)

    Parameters:
    p: probability (scalar or array)
    base: logarithm base ('e', 2, or 10)
    """
    # Avoid log(0) issues
    p = np.clip(p, 1e-10, 1 - 1e-10)

    if base == 'e':
        log_func = np.log
    elif base == 2:
        log_func = np.log2
    elif base == 10:
        log_func = np.log10
    else:
        # General base using change of base formula
        log_func = lambda x: np.log(x) / np.log(base)

    return -p * log_func(p) - (1 - p) * log_func(1 - p)

# Create p values (probability from 0 to 1)
p = np.linspace(0.001, 0.999, 1000)

# Calculate entropy for different bases
H_e = bernoulli_entropy(p, base='e')
H_2 = bernoulli_entropy(p, base=2)
H_10 = bernoulli_entropy(p, base=10)

# The maximum ALWAYS occurs at p = 0.5 for Bernoulli entropy
p_max = 0.5
H_max_e = bernoulli_entropy(p_max, base='e')
H_max_2 = bernoulli_entropy(p_max, base=2)
H_max_10 = bernoulli_entropy(p_max, base=10)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the three entropy functions
plt.plot(p, H_e, 'b-', linewidth=2.5, label=r'$H(p) = -p \ln(p) - (1-p) \ln(1-p)$ (base e)')
plt.plot(p, H_2, 'r-', linewidth=2.5, label=r'$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$ (base 2)')
plt.plot(p, H_10, 'g-', linewidth=2.5, label=r'$H(p) = -p \log_{10}(p) - (1-p) \log_{10}(1-p)$ (base 10)')

# Highlight maximum points (all at p = 0.5)
plt.plot(p_max, H_max_e, 'bo', markersize=10, label=f'Max at $p = 0.5$ (base e)')
plt.plot(p_max, H_max_2, 'rs', markersize=10, label=f'Max at $p = 0.5$ (base 2)')
plt.plot(p_max, H_max_10, 'g^', markersize=10, label=f'Max at $p = 0.5$ (base 10)')

# Add vertical line at p = 0.5
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

# Add grid
plt.grid(True, alpha=0.3)

# Set labels and title
plt.xlabel('Probability p', fontsize=14)
plt.ylabel('Entropy H(p)', fontsize=14)
plt.title('Bernoulli Entropy: $H(p) = -p \\log_b(p) - (1-p) \\log_b(1-p)$', fontsize=16)

# Set axis limits
plt.xlim(0, 1)
plt.ylim(0, max(H_max_e, H_max_2, H_max_10) * 1.1)

# Add legend
plt.legend(fontsize=11, loc='upper center')

# Add text annotations
plt.text(0.08, H_max_e * 0.92, f'Max (base e): {H_max_e:.4f}', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.text(0.08, H_max_2 * 0.85, f'Max (base 2): {H_max_2:.4f}', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
plt.text(0.08, H_max_10 * 0.78, f'Max (base 10): {H_max_10:.4f}', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Add annotation about symmetry
plt.annotate('Symmetric about p=0.5\nMaximum uncertainty', xy=(0.5, H_max_e),
             xytext=(0.65, H_max_e * 0.7),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.4))

# Show the plot
plt.tight_layout()
plt.savefig('bernoulli_entropy.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("伯努利熵 (Bernoulli Entropy): H(p) = -p * log_b(p) - (1-p) * log_b(1-p)")
print("=" * 70)
print(f"\n以 e 为底 (自然对数, nats):")
print(f"  最大值: {H_max_e:.4f} 出现在 p = {p_max}")
print(f"  公式: H(p) = -p * ln(p) - (1-p) * ln(1-p)")

print(f"\n以 2 为底 (bits):")
print(f"  最大值: {H_max_2:.4f} 出现在 p = {p_max}")
print(f"  公式: H(p) = -p * log₂(p) - (1-p) * log₂(1-p)")

print(f"\n以 10 为底 (Hartleys):")
print(f"  最大值: {H_max_10:.4f} 出现在 p = {p_max}")
print(f"  公式: H(p) = -p * log₁₀(p) - (1-p) * log₁₀(1-p)")

print("\n" + "=" * 70)
print("关键观察:")
print("=" * 70)
print("1. 所有函数都在 p = 0.5 处取得最大值（与底数无关！）")
print("2. 当 p = 0.5 时，不确定性最大（抛硬币概率各50%）")
print("3. 当 p → 0 或 p → 1 时，熵 → 0（确定性最大）")
print("4. 函数关于 p = 0.5 对称")
print("5. 底数越大，熵值越小（因为换底公式中的 ln(base) 是分母）")
print("=" * 70)

print("\n物理意义:")
print("- H(p) 衡量二分类结果的不确定性")
print("- p = 0.5: 完全随机，无法预测（熵最大）")
print("- p = 0 或 1: 完全确定，结果可预测（熵为0）")
print("- 在机器学习中，我们希望最小化交叉熵损失")
print("=" * 70)

print("\nPlot saved as 'bernoulli_entropy.png'")
