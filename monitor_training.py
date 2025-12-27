import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """原函数：e^x - log(x)"""
    return np.exp(x) - np.log(x)

def df(x):
    """导数函数：e^x - 1/x"""
    return np.exp(x) - 1/x

def gradient_descent(initial_x, learning_rate, max_iter, tolerance=1e-6):
    """
    带监控的梯度下降
    """
    x = initial_x
    losses = []      # 记录损失函数值
    gradients = []   # 记录梯度值
    x_history = []   # 记录x的变化

    for i in range(max_iter):
        # 计算梯度（迭代需要）
        grad = df(x)

        # 计算损失（监控需要）
        loss = f(x)

        # 记录历史
        losses.append(loss)
        gradients.append(grad)
        x_history.append(x)

        # 检查收敛
        if abs(grad) < tolerance:
            print(f"收敛于第 {i} 次迭代")
            break

        # 更新参数
        x = x - learning_rate * grad

        # 防止x进入无效域（log定义域要求x>0）
        if x <= 0:
            print("警告：x进入无效域，停止训练")
            break

    return x, losses, gradients, x_history

# 执行梯度下降
initial_x = 0.5
learning_rate = 0.01
x_opt, losses, gradients, x_history = gradient_descent(
    initial_x, learning_rate, max_iter=1000
)

print(f"最优解: x = {x_opt:.6f}")
print(f"最小值: f(x) = {f(x_opt):.6f}")
print(f"最终梯度: f'(x) = {df(x_opt):.6f}")

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 损失函数曲线（最重要！）
axes[0, 0].plot(losses)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss (f(x))')
axes[0, 0].set_title('Loss Curve')
axes[0, 0].grid(True)
axes[0, 0].set_yscale('log')  # 对数坐标更清晰

# 2. 梯度曲线
axes[0, 1].plot(gradients, label='Gradient')
axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Zero')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Gradient (df/dx)')
axes[0, 1].set_title('Gradient Curve')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. 参数x的变化轨迹
axes[1, 0].plot(x_history)
axes[1, 0].axhline(y=x_opt, color='r', linestyle='--', label=f'Optimal x={x_opt:.4f}')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('x value')
axes[1, 0].set_title('Parameter Trajectory')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. 函数曲线和优化路径
x_range = np.linspace(0.1, 2, 100)
y_range = f(x_range)
axes[1, 1].plot(x_range, y_range, 'b-', label='f(x) = e^x - log(x)')
axes[1, 1].scatter(x_history, [f(x) for x in x_history],
                   c=np.arange(len(x_history)), cmap='viridis',
                   s=20, label='Optimization Path')
axes[1, 1].scatter(x_opt, f(x_opt), c='red', s=100, marker='*',
                   label=f'Minimum', zorder=5)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('f(x)')
axes[1, 1].set_title('Optimization Visualization')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('gradient_descent_monitoring.png', dpi=100)
print("\n图表已保存为 gradient_descent_monitoring.png")
plt.show()
