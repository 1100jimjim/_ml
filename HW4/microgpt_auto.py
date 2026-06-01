import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        # 計算 Q, K, V
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 注意力矩陣相乘
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        
        # 輸出轉換
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MicroGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        x = self.token_emb(idx)
        x = self.attn(x)
        logits = self.ln(x)
        return logits

# ================= 測試與訓練區塊 =================
vocab_size = 100
n_embd = 64
n_head = 4
batch_size = 32
seq_len = 10

# 1. 建立模型與損失函數
model = MicroGPT(vocab_size, n_embd, n_head)
criterion = nn.CrossEntropyLoss()

# 2. 生成完全對齊的測試資料 (維度都是 batch_size, seq_len)
inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

# 3. 前向傳播
outputs = model(inputs)

# 4. 計算 Loss (將 outputs 壓平對齊 vocab_size，targets 攤平成 1D)
loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

print("========================================")
print(f"🎉 恭喜！模型執行成功！Loss: {loss.item():.4f}")
print("========================================")
