"""
绘制对数伯努利分布函数 log[p^7 * (1-p)^3]
这是对数似然函数
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建p的取值范围 (0.01到0.99，避免log(0))
p = np.linspace(0.01, 0.99, 1000)

# 原始概率函数
f_original = p**7 * (1-p)**3
binomial_original = comb(10, 7) * p**7 * (1-p)**3

# 对数概率函数
log_f = 7 * np.log(p) + 3 * np.log(1-p)
log_binomial = np.log(comb(10, 7)) + 7 * np.log(p) + 3 * np.log(1-p)

# 创建图形
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 原始概率 p^7(1-p)^3
ax1.plot(p, f_original, 'b-', linewidth=2, label=r'$f(p) = p^7(1-p)^3$')
ax1.fill_between(p, f_original, alpha=0.3, color='blue')
ax1.set_xlabel('p', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title(r'Original: $p^7(1-p)^3$', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)
ax1.set_xlim(0, 1)

# 图2: 对数概率 log[p^7(1-p)^3]
ax2.plot(p, log_f, 'r-', linewidth=2, label=r'$\log[f(p)] = 7\log(p) + 3\log(1-p)$')
ax2.fill_between(p, log_f, alpha=0.3, color='red')
ax2.set_xlabel('p', fontsize=12)
ax2.set_ylabel('Log Probability', fontsize=12)
ax2.set_title(r'Log: $\log[p^7(1-p)^3]$', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)
ax2.set_xlim(0, 1)

# 标记最大值点
max_idx2 = np.argmax(log_f)
p_max2 = p[max_idx2]
log_f_max = log_f[max_idx2]
ax2.plot(p_max2, log_f_max, 'ko', markersize=8)
ax2.annotate(f'Maximum: p={p_max2:.3f}\nlog(f)={log_f_max:.3f}',
             xy=(p_max2, log_f_max),
             xytext=(p_max2+0.15, log_f_max),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 图3: 原始完整二项分布概率
ax3.plot(p, binomial_original, 'g-', linewidth=2, label=r'$P(X=7) = \binom{10}{7} p^7(1-p)^3$')
ax3.fill_between(p, binomial_original, alpha=0.3, color='green')
ax3.set_xlabel('p', fontsize=12)
ax3.set_ylabel('Probability', fontsize=12)
ax3.set_title(r'Original Binomial: $P(X=7|n=10, p)$', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=12)
ax3.set_xlim(0, 1)

# 图4: 对数似然函数
ax4.plot(p, log_binomial, 'm-', linewidth=2, label=r'$\log L(p) = \log\binom{10}{7} + 7\log(p) + 3\log(1-p)$')
ax4.fill_between(p, log_binomial, alpha=0.3, color='magenta')
ax4.set_xlabel('p', fontsize=12)
ax4.set_ylabel('Log Likelihood', fontsize=12)
ax4.set_title(r'Log Likelihood: $\log[P(X=7|n=10, p)]$', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=12)
ax4.set_xlim(0, 1)

# 标记最大值点
max_idx4 = np.argmax(log_binomial)
p_max4 = p[max_idx4]
log_binomial_max = log_binomial[max_idx4]
ax4.plot(p_max4, log_binomial_max, 'ko', markersize=8)
ax4.annotate(f'Maximum: p={p_max4:.3f}\nlog(L)={log_binomial_max:.3f}',
             xy=(p_max4, log_binomial_max),
             xytext=(p_max4+0.15, log_binomial_max),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('log_bernoulli_distribution.png', dpi=300, bbox_inches='tight')
print("图形已保存为 log_bernoulli_distribution.png")

print(f"\n对数概率函数 log[p^7(1-p)^3] 的最大值:")
print(f"  - 在 p = {p_max2:.4f} 处取得最大值")
print(f"  - 最大值为 {log_f_max:.6f}")

print(f"\n对数似然函数 log[P(X=7|n=10,p)] 的最大值:")
print(f"  - 在 p = {p_max4:.4f} 处取得最大值")
print(f"  - 最大值为 {log_binomial_max:.6f}")

print(f"\n验证：exp({log_f_max:.6f}) = {np.exp(log_f_max):.6f}")
print(f"原始概率最大值 = {f_original[max_idx2]:.6f}")

plt.show()
