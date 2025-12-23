import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建 x 的范围
x = np.linspace(-3, 3, 400)

# 计算 y = e^x
y = np.exp(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(x, y, 'b-', linewidth=2, label=r'$y = e^x$')

# 添加坐标轴
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# 添加网格
plt.grid(True, alpha=0.3)

# 设置标签和标题
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('指数函数 y = e^x', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 设置坐标轴范围
plt.xlim(-3, 3)
plt.ylim(-0.5, 20)

# 添加一些关键点的标注
plt.plot(0, 1, 'ro', markersize=8)  # (0, 1)
plt.plot(1, np.e, 'ro', markersize=8)  # (1, e)
plt.plot(-1, 1/np.e, 'ro', markersize=8)  # (-1, 1/e)

plt.annotate('(0, 1)', xy=(0, 1), xytext=(0.5, 2),
             fontsize=10, arrowprops=dict(arrowstyle='->', lw=1))
plt.annotate(f'(1, {np.e:.2f})', xy=(1, np.e), xytext=(1.5, 4),
             fontsize=10, arrowprops=dict(arrowstyle='->', lw=1))
plt.annotate('(-1, 1/e)', xy=(-1, 1/np.e), xytext=(-2.5, 1),
             fontsize=10, arrowprops=dict(arrowstyle='->', lw=1))

# 保存图像
plt.savefig('exponential_function.png', dpi=300, bbox_inches='tight')
print("图像已保存为 exponential_function.png")

plt.show()
