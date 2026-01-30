import numpy as np

# 1. 模拟数据：3个特征，1个输出
# 真实规律是非线性的：y = (x1^2 + x2*0.5 - x3) + 噪声
np.random.seed(42)
X = np.random.rand(500, 3)  # 500个样本，每个样本3个特征
y_true = (X[:, 0]**2 + X[:, 1]*0.5 - X[:, 2]).reshape(-1, 1) + np.random.randn(500, 1) * 0.05

# 2. 网络参数初始化
input_size = 3
hidden_size = 10  # 隐藏层有10个神经元  
output_size = 1
lr = 0.01 # 学习率

# 第一层权重 (10x3) 和偏置 (10x1)
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
# 第二层权重 (1x10) 和偏置 (1x1)
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

# 3. 训练循环
for epoch in range(2000):
    # --- 前向传播 ---
    # 输入需要转置为 (3, 500) 适配矩阵运算
    A0 = X.T #(3, 500)

    #w1(10,3)  x  A0(3,500) = Z1(10,500)

    # 第一层计算
    Z1 = np.dot(W1, A0) + b1 # (10, 500)
    A1 = np.maximum(0, Z1)  # ReLU 激活 # (10, 500)
    
    # 第二层计算

    #w2(1,10)  x  A1(10,500) = Z2(1,500)

    Z2 = np.dot(W2, A1) + b2 # (1, 500)
    A2 = Z2  # 最终输出
    
    # 计算 Loss (MSE)
    loss = np.mean((A2.T - y_true)**2)
    
    # --- 反向传播 (追责时刻) ---
    m = X.shape[0]
    dZ2 = A2 - y_true.T
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # 这一步体现了链式法则：经过 ReLU 的导数
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > 0) # ReLU 的导数：正数则传导，负数则为0
    dW1 = (1/m) * np.dot(dZ1, A0.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    # 更新参数
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\n训练完成！现在的模型已经学会了如何处理 3 维特征并进行非线性思考。")