import numpy as np

# 1. 模拟数据：假设真实规律是 y = 3x + 2
# 我们在数据里加一点点随机噪音
X = np.linspace(0, 10, 100).reshape(-1, 1) # 100个输入
y_true = 3 * X + 2 + np.random.randn(100, 1) * 0.5

# 2. 初始化参数（瞎猜）
w = np.random.randn(1) # 随机给个权重
b = np.zeros(1)        # 偏置设为0
learning_rate = 0.01

print(f"训练前：w={w[0]:.2f}, b={b[0]:.2f}")

# 3. 训练循环（转动旋钮）
for epoch in range(100):
    # 前向传播：计算预测值
    print (f"Epoch {epoch+1}: w={w}, b={b}")
    y_pred = w * X + b

    
    # 计算损失 (MSE)
    loss = np.mean((y_pred - y_true)**2)
    
    # 反向传播：计算梯度 (偏导数)
    # dw = dLoss/dw, db = dLoss/db
    dw = np.mean(2 * X * (y_pred - y_true))
    db = np.mean(2 * (y_pred - y_true))
    
    # 更新参数（逆着梯度走）
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss:.4f}, w: {w[0]:.2f}, b: {b[0]:.2f}")

print(f"\n训练结束：w 应接近 3，b 应接近 2")