import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """标准的 sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """sigmoid 的导数: σ'(x) = σ(x)(1-σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

# 生成 x 值范围
x = np.linspace(-10, 10, 1000)
y = sigmoid(x)
dy = sigmoid_derivative(x)

# 创建图像
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ============ 左上：sigmoid 函数 ============
axes[0, 0].plot(x, y, 'b-', linewidth=2, label='σ(x) = 1/(1+e⁻ˣ)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', linewidth=1)
axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=1)
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('σ(x)', fontsize=12)
axes[0, 0].set_title('Sigmoid Function', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].set_ylim(-0.1, 1.1)

# 标注饱和区域
axes[0, 0].axvspan(5, 10, alpha=0.2, color='red', label='Saturation Region')
axes[0, 0].axvspan(-10, -5, alpha=0.2, color='red')
axes[0, 0].text(7, 0.5, 'Saturated\n(Gradient ≈ 0)', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
axes[0, 0].text(-7, 0.5, 'Saturated\n(Gradient ≈ 0)', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
axes[0, 0].text(0, 0.2, 'Active Region\n(Large Gradient)', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))

# ============ 右上：sigmoid 的导数 ============
axes[0, 1].plot(x, dy, 'g-', linewidth=2, label="σ'(x) = σ(x)(1-σ(x))")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel("σ'(x)", fontsize=12)
axes[0, 1].set_title('Sigmoid Derivative (Gradient)', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].set_ylim(-0.05, 0.3)

# 标注最大值
max_grad_idx = np.argmax(dy)
axes[0, 1].scatter([x[max_grad_idx]], [dy[max_grad_idx]], c='red', s=100, marker='*', zorder=5)
axes[0, 1].text(x[max_grad_idx], dy[max_grad_idx] + 0.02,
                f'Max gradient = {dy[max_grad_idx]:.4f}\n at x = {x[max_grad_idx]:.2f}',
                fontsize=10, ha='center')

# 标注梯度接近0的区域
axes[0, 1].axvspan(5, 10, alpha=0.2, color='red')
axes[0, 1].axvspan(-10, -5, alpha=0.2, color='red')
axes[0, 1].text(7, 0.05, 'Vanishing\nGradient', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

# ============ 左下：不同点的sigmoid值 ============
test_points = [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]
sigmoid_values = [sigmoid(val) for val in test_points]
grad_values = [sigmoid_derivative(val) for val in test_points]

bars = axes[1, 0].bar(range(len(test_points)), sigmoid_values,
                       color=['red' if abs(val) > 5 else 'green' for val in test_points],
                       alpha=0.7)
axes[1, 0].axhline(y=0.5, color='r', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('σ(x)', fontsize=12)
axes[1, 0].set_title('Sigmoid Values at Different x', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(range(len(test_points)))
axes[1, 0].set_xticklabels(test_points)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 添加数值标注
for i, (bar, val, grad) in enumerate(zip(bars, sigmoid_values, grad_values)):
    height = bar.get_height()
    if abs(test_points[i]) <= 3:
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}\n∇={grad:.3f}',
                       ha='center', va='bottom', fontsize=8)

# ============ 右下：梯度值柱状图 ============
bars_grad = axes[1, 1].bar(range(len(test_points)), grad_values,
                           color=['red' if val < 0.01 else 'green' for val in grad_values],
                           alpha=0.7)
axes[1, 1].axhline(y=0.01, color='orange', linestyle='--', linewidth=2,
                  label='Threshold = 0.01')
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel("σ'(x)", fontsize=12)
axes[1, 1].set_title('Gradient Values at Different x', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(len(test_points)))
axes[1, 1].set_xticklabels(test_points)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 添加数值标注
for i, (bar, val) in enumerate(zip(bars_grad, grad_values)):
    height = bar.get_height()
    if val > 0.01:  # 只标注有意义的梯度
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=8)
    else:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., 0.0005,
                       f'≈0',
                       ha='center', va='bottom', fontsize=8, color='red')

plt.tight_layout()
plt.savefig('sigmoid_saturation_problem.png', dpi=100, bbox_inches='tight')
print("图表已保存为 sigmoid_saturation_problem.png")

# 打印详细分析
print("\n" + "="*80)
print("SIGMOID 函数的饱和问题分析")
print("="*80)

print("\n1. 函数值分析:")
print("-"*80)
for x_val in test_points:
    sig_val = sigmoid(x_val)
    grad_val = sigmoid_derivative(x_val)
    status = "✓ 活跃区" if abs(grad_val) > 0.01 else "✗ 饱和区"
    print(f"  x = {x_val:3d}: σ(x) = {sig_val:.6f},  σ'(x) = {grad_val:.6f}  {status}")

print("\n2. 关键观察:")
print("-"*80)
print("  • 在 |x| > 5 时，sigmoid 输出接近 0 或 1")
print(f"  • x = 5:  σ(5) = {sigmoid(5):.6f},  σ'(5) = {sigmoid_derivative(5):.6f}")
print(f"  • x = 10: σ(10) = {sigmoid(10):.8f}, σ'(10) = {sigmoid_derivative(10):.8f}")
print("  • 导数接近 0 意味着梯度消失，神经网络难以学习")

print("\n3. 对神经网络的影响:")
print("-"*80)
print("  ✗ 问题：梯度消失 (Vanishing Gradient Problem)")
print("    - 当 x 很大或很小时，梯度 ≈ 0")
print("    - 深层网络的梯度连乘，导致梯度趋于 0")
print("    - 参数更新几乎停止，网络无法学习")
print("")
print("  ✓ 解决方案：")
print("    - ReLU: f(x) = max(0, x)，梯度恒为 0 或 1")
print("    - Leaky ReLU: 允许小的负梯度")
print("    - ELU, SELU, GELU 等现代激活函数")

print("\n4. 为什么现代深度学习很少用 Sigmoid?")
print("-"*80)
print("  • 饱和区域宽：只有 [-3, 3] 区间梯度较大")
print("  • 梯度小：最大梯度仅为 0.25")
print("  • 输出不以 0 为中心：会导致后续层的输入偏移")
print("  • 计算成本高：需要计算指数函数")
print("")
print("  相比之下，ReLU 计算简单（max操作），梯度恒定，不会饱和（正区域）")
print("="*80)
