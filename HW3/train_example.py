import random
# 假設這些是從你的 nn0.py 中匯入的核心組件
from nn0 import Value, Adam, linear, rmsnorm, softmax

# ==========================================
# 1. 超參數與參數初始化 (Hyperparameters & Initialization)
# ==========================================
vocab_size = 4  # 假設我們的詞表只有 4 個單字 (Token ID: 0, 1, 2, 3)
d_model = 8     # 隱藏層/特徵維度大小

# 初始化模型權重 (使用 Value 物件來追蹤梯度)
# W1: [d_model, d_model] - 模擬第一層權重
W1 = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d_model)] for _ in range(d_model)]
# W2: [d_model, vocab_size] - 模擬輸出層權重 (Logits)
W2 = [[Value(random.uniform(-0.1, 0.1)) for _ in range(vocab_size)] for _ in range(d_model)]

# 將所有權重展平，交給 Adam 優化器管理
all_parameters = [w for row in W1 for w in row] + [w for row in W2 for w in row]
optimizer = Adam(parameters=all_parameters, lr=0.01)

# ==========================================
# 2. 定義前向傳播 (Forward Pass)
# ==========================================
def model_forward(x_input):
    """
    結合 nn0.py 的現代化 Transformer 算子建構模型
    """
    # 第一層：線性變換 (W1 * x)
    h = linear(x_input, W1)
    
    # 進行 RMSNorm 正規化 (提升大模型訓練的數值穩定性)
    h_norm = rmsnorm(h)
    
    # 輸出層：映射到詞表大小 (Logits)
    logits = linear(h_norm, W2)
    
    # Softmax：將 Logits 轉換為機率分佈
    probs = softmax(logits)
    
    return probs

# ==========================================
# 3. 模型訓練循環 (Training Loop / 類似 gd() 的行為)
# ==========================================
def train_step(x_input, target_id, step, num_steps):
    # a. 前向傳播 (Forward Pass)
    probs = model_forward(x_input)
    
    # b. 計算損失 (Negative Log-Likelihood)
    # 取出目標 Token 的預測機率，並加上 .log() 計算負對數似然
    target_prob = probs[target_id]
    loss = -target_prob.log() 
    
    # c. 梯度清零 (Zero Gradients)
    # 在反向傳播前，必須清空上一步累積的梯度
    optimizer.zero_grad()
    
    # d. 反向傳播 (Backward Pass)
    # 這裡會觸發 Value 類別的自動微分，利用拓撲排序與鏈式法則計算所有參數的梯度
    loss.backward()
    
    # 模擬學習率線性衰減 (Linear Learning Rate Decay)
    current_lr = optimizer.lr * (1 - step / num_steps)
    
    # e. 更新權重 (Update)
    # 使用 Adam 優化器 (結合動量 m 與速度 v) 更新 W1 與 W2
    optimizer.step(lr=current_lr)
    
    return loss, target_prob

# ==========================================
# 4. 執行測試
# ==========================================
# 模擬一個隨機的輸入特徵向量
x_example = [Value(random.random()) for _ in range(d_model)]
# 假設正確答案是 Token ID = 2
target_token_id = 2 

num_epochs = 50

print("開始訓練...\n")
for epoch in range(num_epochs):
    # 由於 nn0.py 是以計算圖為基礎，每次 forward 都需要重新建立 Value 節點的關係
    # 在實際框架中，x_example 的 data 不變，但會重新參與運算
    loss_val, prob_val = train_step(x_example, target_token_id, step=epoch, num_steps=num_epochs)
    
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:02d} | Loss: {loss_val.data:.4f} | Target Token 機率: {prob_val.data:.4f}")

print("\n訓練完成！你可以看到目標 Token 的機率逐漸逼近 1.0。")
