"""
绘制伯努利分布函数 p^7 * (1-p)^3
这是二项分布 B(n=10, k=7) 中特定结果组合的概率项
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建p的取值范围 (0到1)
p = np.linspace(0, 1, 1000)

# 计算概率函数 f(p) = p^7 * (1-p)^3
# 这是二项分布 B(10, 7) 中特定结果的概率项（不考虑组合数）
f = p**7 * (1-p)**3

# 计算完整的二项分布概率（包含组合数 C(10,7)=120）
from scipy.special import comb
binomial_prob = comb(10, 7) * p**7 * (1-p)**3

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 图1: 单项概率函数 p^7(1-p)^3
ax1.plot(p, f, 'b-', linewidth=2, label=r'$f(p) = p^7(1-p)^3$')
ax1.fill_between(p, f, alpha=0.3, color='blue')
ax1.set_xlabel('p', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title(r'Bernoulli Distribution Term: $p^7(1-p)^3$', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, max(f) * 1.1)

# 标记最大值点
max_idx = np.argmax(f)
p_max = p[max_idx]
f_max = f[max_idx]
ax1.plot(p_max, f_max, 'ro', markersize=8, label=f'Max at p={p_max:.3f}')
ax1.annotate(f'Maximum: p={p_max:.3f}\nf(p)={f_max:.6f}',
             xy=(p_max, f_max),
             xytext=(p_max+0.15, f_max),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 图2: 完整的二项分布概率 B(10, 7, p)
ax2.plot(p, binomial_prob, 'g-', linewidth=2, label=r'$P(X=7) = \binom{10}{7} p^7(1-p)^3$')
ax2.fill_between(p, binomial_prob, alpha=0.3, color='green')
ax2.set_xlabel('p', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title(r'Binomial Distribution: $P(X=7|n=10, p)$', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, max(binomial_prob) * 1.1)

# 标记最大值点
max_idx2 = np.argmax(binomial_prob)
p_max2 = p[max_idx2]
binomial_max = binomial_prob[max_idx2]
ax2.plot(p_max2, binomial_max, 'ro', markersize=8)
ax2.annotate(f'Maximum: p={p_max2:.3f}\nProbability={binomial_max:.6f}',
             xy=(p_max2, binomial_max),
             xytext=(p_max2+0.15, binomial_max),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('bernoulli_distribution.png', dpi=300, bbox_inches='tight')
print("图形已保存为 bernoulli_distribution.png")
print(f"\n函数 p^7(1-p)^3 的最大值:")
print(f"  - 在 p = {p_max:.4f} 处取得最大值")
print(f"  - 最大值为 {f_max:.6f}")
print(f"\n完整二项分布概率 (C(10,7)*p^7*(1-p)^3):")
print(f"  - 在 p = {p_max2:.4f} 处取得最大值")
print(f"  - 最大概率为 {binomial_max:.6f}")

plt.show()
