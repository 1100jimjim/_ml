import random
# 引入你提供原汁原味的 nn0 框架
from nn0 import Value, Adam, linear, rmsnorm, softmax

# ==========================================
# 1. 超參數與參數初始化
# ==========================================
vocab_size = 4  
d_model = 8     

# 初始化模型權重
W1 = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d_model)] for _ in range(d_model)]
W2 = [[Value(random.uniform(-0.1, 0.1)) for _ in range(vocab_size)] for _ in range(d_model)]

# 展開參數
all_parameters = [w for row in W1 for w in row] + [w for row in W2 for w in row]

# 根據 nn0.py，第一個參數名稱是 params，我們直接放入清單
optimizer = Adam(all_parameters, lr=0.01)

# ==========================================
# 2. 定義前向傳播 (Forward Pass)
# ==========================================
def model_forward(x_input):
    h = linear(x_input, W1)
    h_norm = rmsnorm(h)
    logits = linear(h_norm, W2)
    probs = softmax(logits)
    return probs

# ==========================================
# 3. 模型訓練步驟 (Training Step)
# ==========================================
def train_step(x_input, target_id, step, num_steps):
    probs = model_forward(x_input)
    
    # 根據 nn0.py 裡 gd 函數的寫法，使用負對數似然
    target_prob = probs[target_id]
    loss = -target_prob.log() 
    
    # 反向傳播 (直接呼叫，因為等一下 step() 會自動清空梯度)
    loss.backward()
    
    # 計算線性衰減的學習率
    current_lr = optimizer.lr * (1 - step / num_steps)
    
    # 根據 nn0.py，正確的參數名稱是 lr_override
    optimizer.step(lr_override=current_lr)
    
    return loss, target_prob

# ==========================================
# 4. 執行測試與訓練迴圈 (Training Loop)
# ==========================================
if __name__ == "__main__":
    x_example = [Value(random.random()) for _ in range(d_model)]
    
    target_token_id = 2 
    num_epochs = 50

    print("=== Start Training Mini Transformer ===\n")
    
    for epoch in range(num_epochs):
        loss_val, prob_val = train_step(x_example, target_token_id, step=epoch, num_steps=num_epochs)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:02d} | Loss: {loss_val.data:.4f} | Target Prob: {prob_val.data:.4f}")

    print("\n=== Training Complete! ===")
