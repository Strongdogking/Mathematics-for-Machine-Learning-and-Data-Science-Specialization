import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义函数 y = (1 + 1/x)^x
def f(x):
    return (1 + 1/x)**x

# 生成x值（避免x=0和负值）
x_positive = np.linspace(0.1, 100, 1000)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：较大范围 (0.1 到 100)
axes[0].plot(x_positive, f(x_positive), 'b-', linewidth=2, label=r'$y = (1 + \frac{1}{x})^x$')
axes[0].axhline(y=np.e, color='r', linestyle='--', label=f'y = e ≈ {np.e:.4f}')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title(r'函数 $y = (1 + \frac{1}{x})^x$ 的图像 (0.1 ≤ x ≤ 100)', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 100])
axes[0].set_ylim([1.5, 3.5])

# 图2：较小范围，更详细地观察趋近于e的过程 (1 到 20)
x_detailed = np.linspace(1, 20, 500)
axes[1].plot(x_detailed, f(x_detailed), 'b-', linewidth=2, label=r'$y = (1 + \frac{1}{x})^x$')
axes[1].axhline(y=np.e, color='r', linestyle='--', label=f'y = e ≈ {np.e:.4f}')

# 标注一些特殊点
special_x = [1, 2, 5, 10]
for sx in special_x:
    sy = f(sx)
    axes[1].plot(sx, sy, 'ro', markersize=6)
    axes[1].annotate(f'({sx}, {sy:.4f})', xy=(sx, sy), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)

axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title(r'函数详细视图 (1 ≤ x ≤ 20)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/d/codework/math_for_ml/function_plot.png', dpi=300, bbox_inches='tight')
print("图像已保存到 function_plot.png")
plt.show()

# 打印一些函数值
print("\n函数值表：")
print("-" * 40)
print(f"{'x':>10} | {'y = (1+1/x)^x':>20}")
print("-" * 40)
for x in [1, 2, 5, 10, 50, 100, 1000]:
    print(f"{x:>10} | {f(x):>20.6f}")
print("-" * 40)
print(f"{'∞':>10} | {np.e:>20.6f} (极限值)")
print("-" * 40)
