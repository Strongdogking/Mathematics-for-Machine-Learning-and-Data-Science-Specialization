"""
比较原始概率和对数概率的导数行为
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建p的取值范围
p = np.linspace(0.01, 0.99, 1000)

# 原始概率函数 f(p) = p^7 * (1-p)^3
f = p**7 * (1-p)**3

# 对数概率函数
log_f = 7 * np.log(p) + 3 * np.log(1-p)

# 计算导数
# 原始概率的导数
f_prime = 7 * p**6 * (1-p)**3 - 3 * p**7 * (1-p)**2

# 对数概率的导数
log_f_prime = 7/p - 3/(1-p)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 原始概率函数
axes[0,0].plot(p, f, 'b-', linewidth=2, label=r'$f(p) = p^7(1-p)^3$')
axes[0,0].fill_between(p, f, alpha=0.3, color='blue')
axes[0,0].set_xlabel('p', fontsize=12)
axes[0,0].set_ylabel('Probability', fontsize=12)
axes[0,0].set_title('Original Probability Function', fontsize=14, fontweight='bold')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend(fontsize=12)
axes[0,0].set_xlim(0, 1)

# 图2: 原始概率的导数
axes[0,1].plot(p, f_prime, 'g-', linewidth=2, label=r"$f'(p)$")
axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[0,1].set_xlabel('p', fontsize=12)
axes[0,1].set_ylabel('Derivative', fontsize=12)
axes[0,1].set_title("Derivative of Original Probability", fontsize=14, fontweight='bold')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend(fontsize=12)
axes[0,1].set_xlim(0, 1)
axes[0,1].set_ylim(-0.01, 0.01)

# 标注p=0.7处的导数
axes[0,1].plot(0.7, 0, 'ro', markersize=8)
axes[0,1].annotate('p=0.7\nf\'(p)=0', xy=(0.7, 0), xytext=(0.5, 0.003),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

# 图3: 对数概率函数
axes[1,0].plot(p, log_f, 'r-', linewidth=2, label=r'$\log[f(p)] = 7\log(p) + 3\log(1-p)$')
axes[1,0].fill_between(p, log_f, alpha=0.3, color='red')
axes[1,0].set_xlabel('p', fontsize=12)
axes[1,0].set_ylabel('Log Probability', fontsize=12)
axes[1,0].set_title('Log Probability Function', fontsize=14, fontweight='bold')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend(fontsize=12)
axes[1,0].set_xlim(0, 1)

# 图4: 对数概率的导数
axes[1,1].plot(p, log_f_prime, 'm-', linewidth=2, label=r"$\frac{d}{dp}\log[f(p)] = \frac{7}{p} - \frac{3}{1-p}$")
axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1,1].set_xlabel('p', fontsize=12)
axes[1,1].set_ylabel('Derivative', fontsize=12)
axes[1,1].set_title("Derivative of Log Probability", fontsize=14, fontweight='bold')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend(fontsize=12)
axes[1,1].set_xlim(0, 1)
axes[1,1].set_ylim(-50, 50)

# 标注p=0.7处的导数
axes[1,1].plot(0.7, 0, 'ro', markersize=8)
axes[1,1].annotate('p=0.7\nd/dp[log f]=0', xy=(0.7, 0), xytext=(0.5, 10),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

plt.tight_layout()
plt.savefig('compare_derivatives.png', dpi=300, bbox_inches='tight')
print("图形已保存为 compare_derivatives.png")

# 打印关键点的导数值
print("\n=== 关键点的导数比较 ===\n")
print("在 p=0.7 (最大值点):")
idx_07 = np.abs(p-0.7).argmin()
print(f"  原始概率导数: {f_prime[idx_07]:.6f}")
print(f"  对数概率导数: {log_f_prime[idx_07]:.6f}")

print("\n在 p→0 时:")
print(f"  原始概率值: {f[0]:.6f}")
print(f"  原始概率导数: {f_prime[0]:.6f}")
print(f"  对数概率导数: {log_f_prime[0]:.6f} → +∞")

print("\n在 p=0.01 时:")
idx = np.abs(p-0.01).argmin()
print(f"  原始概率值: {f[idx]:.6f}")
print(f"  原始概率导数: {f_prime[idx]:.8f}")
print(f"  对数概率导数: {log_f_prime[idx]:.2f}")

plt.show()
