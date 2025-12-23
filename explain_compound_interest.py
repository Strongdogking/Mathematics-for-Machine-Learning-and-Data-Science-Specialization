import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示（尝试多种字体）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1：复利次数 vs 最终收益
n_values = [1, 2, 4, 12, 52, 365, 365*24, 365*24*60]
values = [(1 + 1/n)**n for n in n_values]
axes[0, 0].bar(range(len(n_values)), values, color='skyblue', alpha=0.7, edgecolor='navy')
axes[0, 0].axhline(y=np.e, color='r', linestyle='--', linewidth=2, label=f'e = {np.e:.6f}')
axes[0, 0].axhline(y=2, color='g', linestyle='--', linewidth=2, label='单利 = 2')
axes[0, 0].set_xlabel('Compounding Frequency', fontsize=12)
axes[0, 0].set_ylabel('Final Amount ($1 invested at 100% APR)', fontsize=12)
axes[0, 0].set_title('Compound Interest: More Frequent = Higher Return', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(len(n_values)))
axes[0, 0].set_xticklabels(['Annual', 'Semi-\nannual', 'Quarterly', 'Monthly',
                               'Weekly', 'Daily', 'Hourly', 'Every\nminute'], rotation=45, fontsize=9)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 标注差值
axes[0, 0].annotate(f'e - 2 = {np.e - 2:.6f}', xy=(1, 2.5), fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 图2：函数趋近于e的过程
x = np.linspace(1, 100, 1000)
y = (1 + 1/x)**x
axes[0, 1].plot(x, y, 'b-', linewidth=2, label='y = (1 + 1/x)^x')
axes[0, 1].axhline(y=np.e, color='r', linestyle='--', linewidth=2, label=f'e = {np.e:.6f}')
axes[0, 1].axhline(y=2, color='g', linestyle='--', linewidth=2, label='Simple Interest = 2')
axes[0, 1].fill_between(x, 2, np.e, alpha=0.2, color='red', label='Compound Interest Bonus')
axes[0, 1].set_xlabel('x (Number of compounding periods)', fontsize=12)
axes[0, 1].set_ylabel('Amount', fontsize=12)
axes[0, 1].set_title('How (1 + 1/x)^x Approaches e', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([0, 100])

# 图3：泰勒级数展示（e-2的组成部分）
terms = ['1/2!', '1/3!', '1/4!', '1/5!', '1/6!', '1/7!', '1/8!']
term_values = [1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320]
cumulative = np.cumsum(term_values)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(terms)))

bars = axes[1, 0].bar(range(len(terms)), term_values, color=colors, alpha=0.7, edgecolor='darkred')
axes[1, 0].plot(range(len(terms)), cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative Sum')
axes[1, 0].axhline(y=np.e - 2, color='b', linestyle='--', linewidth=2, label=f'Total = e - 2 = {np.e - 2:.6f}')
axes[1, 0].set_xlabel('Terms in Taylor Series', fontsize=12)
axes[1, 0].set_ylabel('Contribution to e - 2', fontsize=12)
axes[1, 0].set_title('Taylor Series: e - 2 = 1/2! + 1/3! + 1/4! + ...', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(range(len(terms)))
axes[1, 0].set_xticklabels(terms, rotation=45, fontsize=9)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 添加数值标注
for i, (bar, val) in enumerate(zip(bars, term_values)):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# 图4：连续增长 vs 离散增长
t = np.linspace(0, 1, 100)
continuous = np.exp(t)
discrete_steps = [0, 0.5, 1]
discrete_values = [1, 1.5, 2]

axes[1, 1].plot(t, continuous, 'b-', linewidth=3, label=f'Continuous: e^t (final = {np.e:.4f})')
axes[1, 1].plot(t, 1 + t, 'orange', linewidth=3, linestyle='--',
               label=f'Simple Interest: 1 + t (final = 2)')
axes[1, 1].fill_between(t, 1 + t, np.exp(t), alpha=0.3, color='red',
                        label=f'Bonus: e^t - (1+t) = {np.e - 2:.4f}')
axes[1, 1].set_xlabel('Time (years)', fontsize=12)
axes[1, 1].set_ylabel('Amount ($1 at 100% APR)', fontsize=12)
axes[1, 1].set_title('Continuous vs Simple Interest Growth', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([1, 3])

plt.tight_layout()
plt.savefig('/mnt/d/codework/math_for_ml/compound_interest_explanation.png', dpi=300, bbox_inches='tight')
print("图像已保存到 compound_interest_explanation.png")
plt.show()

# 打印数学解释
print("\n" + "="*70)
print("e - 2 ≈ 0.71828 的数学含义")
print("="*70)
print("\n💰 复利解释：")
print("  本金: $1")
print("  年利率: 100%")
print(f"  单利（1年结算1次）: ${2:.6f}")
print(f"  连续复利: ${np.e:.6f}")
print(f"  差值（复利红利）: ${np.e - 2:.6f}")

print("\n📐 数学表示：")
print("  方法1 - 泰勒级数：")
print(f"    e - 2 = 1/2! + 1/3! + 1/4! + ... = {np.e - 2:.6f}")
print("    展开式：")
import math
for i in range(7):
    factorial = math.factorial(i+2)
    term = 1/factorial
    print(f"      1/{i+2}! = {term:.6f}")

print("\n  方法2 - 极限定义：")
print("    e - 2 = lim(n→∞)[(1 + 1/n)^n - 2]")

print("\n  方法3 - 积分形式：")
print(f"    e - 2 = ∫₀¹(e^x - 1)dx = {np.e - 2:.6f}")

print("\n🎯 直观理解：")
print("  这个差值代表了当利息结算频率无限增加时，")
print("  由于'利息产生利息'的效应而累积的额外收益。")
print("="*70)
