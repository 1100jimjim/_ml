import numpy as np
import matplotlib.pyplot as plt

# 1. 全連接層 (Dense Layer)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# 2. ReLU 激勵函數
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# 3. Softmax 激勵函數
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# 4. 損失函數 (Categorical Cross-Entropy)
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# 5. 結合 Softmax 與 Cross-Entropy 的反向傳播模組
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# 6. 隨機梯度下降優化器 (SGD Optimizer)
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        
    def update_parameters(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

# ==========================================
# 專案實戰：賽事量化預測模型訓練
# ==========================================

# 1. 建立一個模擬的賽事資料集
def create_sports_data(samples=1000):
    np.random.seed(42)
    X = np.random.randn(samples, 5)
    y = np.zeros(samples, dtype=int)
    
    for i in range(samples):
        power_diff = X[i, 0] * 1.5 - X[i, 1] * 1.5 + X[i, 4] * 0.8
        if power_diff > 0.8:
            y[i] = 0 
        elif power_diff < -0.8:
            y[i] = 2 
        else:
            y[i] = 1 
    return X, y

X_train, y_train = create_sports_data(1000)

# 2. 初始化網路結構與優化器
layer1 = Layer_Dense(5, 16)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(16, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.8)

# --- 準備記錄訓練軌跡的容器 ---
epoch_history = []
loss_history = []
accuracy_history = []

print("--- Engine Started: Sports Prediction Model Training ---")

# 3. 執行訓練迴圈
for epoch in range(101):
    # --- 前向傳播 ---
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    
    # --- 計算 Loss 與 Accuracy ---
    loss_activation.activation.forward(layer2.output)
    loss = loss_activation.loss.calculate(loss_activation.activation.output, y_train)
    
    predictions = np.argmax(loss_activation.activation.output, axis=1)
    accuracy = np.mean(predictions == y_train)
    
    # --- 反向傳播 ---
    loss_activation.backward(loss_activation.activation.output, y_train)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    
    # --- 更新權重 ---
    optimizer.update_parameters(layer1)
    optimizer.update_parameters(layer2)
    
    # --- 記錄歷史數據 ---
    epoch_history.append(epoch)
    loss_history.append(loss)
    accuracy_history.append(accuracy)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")

# ==========================================
# 產生報告用圖表 (Matplotlib 視覺化)
# ==========================================
plt.style.use('dark_background') # 使用深色背景讓圖表看起來更專業
fig, ax1 = plt.subplots(figsize=(10, 6))

# 繪製 Loss 曲線 (紅色)
color = 'tab:red'
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', color=color, fontsize=12)
ax1.plot(epoch_history, loss_history, color=color, linewidth=2, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

# 建立共用 X 軸的第二個 Y 軸，用來繪製 Accuracy 曲線 (綠色)
ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel('Accuracy', color=color, fontsize=12)  
ax2.plot(epoch_history, accuracy_history, color=color, linewidth=2, linestyle='--', label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# 設定圖表標題與排版
plt.title('Neural Network Engine Training Performance', fontsize=16, fontweight='bold')
fig.tight_layout()  

# 顯示圖表
plt.show()
