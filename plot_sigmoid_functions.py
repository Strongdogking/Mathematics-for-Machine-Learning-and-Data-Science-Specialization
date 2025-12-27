import numpy as np
import matplotlib.pyplot as plt

# 定义两个函数
def sigmoid1(x):
    """y = 1/(1+e^-x) 标准的 sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid2(x):
    """y = 1/(1+e^x) 另一个 S 形函数"""
    return 1 / (1 + np.exp(x))

# 生成 x 值范围
x = np.linspace(-10, 10, 1000)

# 计算 y 值
y1 = sigmoid1(x)
y2 = sigmoid2(x)

# 创建图像
plt.figure(figsize=(12, 6))

# 绘制第一个函数
plt.subplot(1, 2, 1)
plt.plot(x, y1, 'b-', linewidth=2, label='y = 1/(1+e⁻ˣ)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='y = 0.5')
plt.axvline(x=0, color='r', linestyle='--', linewidth=1, label='x = 0')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('y = 1/(1+e⁻ˣ) - Sigmoid 函数', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.ylim(-0.1, 1.1)

# 添加关键点标注
plt.text(2, 0.8, '当 x→∞, y→1', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
plt.text(-7, 0.2, '当 x→-∞, y→0', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
plt.text(0.5, 0.45, '(0, 0.5)\n对称中心', fontsize=9, ha='center')

# 绘制第二个函数
plt.subplot(1, 2, 2)
plt.plot(x, y2, 'g-', linewidth=2, label='y = 1/(1+eˣ)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='y = 0.5')
plt.axvline(x=0, color='r', linestyle='--', linewidth=1, label='x = 0')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('y = 1/(1+eˣ) - 倒置 Sigmoid', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.ylim(-0.1, 1.1)

# 添加关键点标注
plt.text(2, 0.2, '当 x→∞, y→0', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
plt.text(-7, 0.8, '当 x→-∞, y→1', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
plt.text(-0.5, 0.45, '(0, 0.5)\n对称中心', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('sigmoid_functions.png', dpi=100, bbox_inches='tight')
print("图表已保存为 sigmoid_functions.png")
plt.show()

# 打印一些关键信息
print("\n" + "="*60)
print("函数特性分析")
print("="*60)
print("\n函数 1: y = 1/(1+e⁻ˣ)")
print("-"*60)
print(f"  当 x = 0:  y = {sigmoid1(0):.4f}")
print(f"  当 x = 1:  y = {sigmoid1(1):.4f}")
print(f"  当 x = -1: y = {sigmoid1(-1):.4f}")
print(f"  当 x→∞:   y → 1")
print(f"  当 x→-∞:  y → 0")
print(f"  导数: dy/dx = y(1-y)")

print("\n函数 2: y = 1/(1+eˣ)")
print("-"*60)
print(f"  当 x = 0:  y = {sigmoid2(0):.4f}")
print(f"  当 x = 1:  y = {sigmoid2(1):.4f}")
print(f"  当 x = -1: y = {sigmoid2(-1):.4f}")
print(f"  当 x→∞:   y → 0")
print(f"  当 x→-∞:  y → 1")
print(f"  导数: dy/dx = -y(1-y)")

print("\n两个函数的关系:")
print("-"*60)
print("  第二个函数是第一个函数关于 y = 0.5 的对称")
print("  即: sigmoid2(x) = 1 - sigmoid1(x)")
print(f"  验证: sigmoid1(1) + sigmoid2(1) = {sigmoid1(1) + sigmoid2(1):.4f}")
