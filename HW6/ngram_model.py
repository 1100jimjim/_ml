import urllib.request
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict, Counter

# ── 參數設定 ──────────────────────────────────────────────
CORPUS_URL = "https://raw.githubusercontent.com/ccc114b/cccocw/main/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92/07-%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B/01-%E5%82%B3%E7%B5%B1%E6%96%B9%E6%B3%95/lm/tw.txt"
CONTEXT_SIZE = 2      # N-gram 的 N-1 (用前 2 個字預測下 1 個字)
MAX_GEN_LEN  = 30     # 最多生成幾個字
TEMPERATURE  = 1.0    # 抽樣溫度
TOP_K        = 5      # 限制抽樣範圍
# ──────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════
#  1. 讀取與處理語料
# ════════════════════════════════════════════════════════════
def load_corpus_from_url(url: str) -> list[str]:
    """從網址讀取語料，每行視為一句，句尾插入 <EOS>"""
    print("正在下載並讀取語料庫...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        text = response.read().decode('utf-8')
    
    chars = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            chars.extend(list(line))
            chars.append("<EOS>")
    return chars

# ════════════════════════════════════════════════════════════
#  2. 建立馬可夫模型 (Markov Chain)
# ════════════════════════════════════════════════════════════
def build_markov_model(chars: list[str], context_size: int):
    """
    建立字典格式的模型：
    model[context_tuple][next_char] = 出現次數
    例如: model[('台', '灣')]['人'] = 5
    """
    print(f"正在建立 {context_size+1}-gram 馬可夫模型...")
    model = defaultdict(Counter)
    
    for i in range(len(chars) - context_size):
        context = tuple(chars[i : i + context_size])
        next_char = chars[i + context_size]
        model[context][next_char] += 1
        
    print(f"模型建立完成！共記錄了 {len(model)} 種不同的上下文組合。\n")
    return model

# ════════════════════════════════════════════════════════════
#  3. 抽樣與生成機制
# ════════════════════════════════════════════════════════════
def sample_next(context: tuple, model: dict, temperature: float = 1.0, top_k: int = 5):
    """根據上下文從模型中抽樣下一個字"""
    if context not in model:
        return "<EOS>"  # 如果遇到沒見過的上下文，直接結束
    
    # 取得下一個字的次數分布
    next_chars_counts = model[context]
    chars = list(next_chars_counts.keys())
    counts = list(next_chars_counts.values())
    
    # 轉為初始機率分布
    proba = np.array(counts) / sum(counts)
    
    # Top-K 篩選
    if top_k and top_k < len(proba):
        top_idx = np.argsort(proba)[::-1][:top_k]
        mask = np.zeros_like(proba)
        mask[top_idx] = proba[top_idx]
        proba = mask
    
    # Temperature Scaling
    log_p = np.log(np.clip(proba, 1e-10, None)) / temperature
    exp_p = np.exp(log_p - np.max(log_p)) # 扣掉 max 以防數值溢位
    exp_p /= exp_p.sum()                  # Softmax 轉回機率
    
    # 根據機率抽樣
    chosen_idx = np.random.choice(len(chars), p=exp_p)
    return chars[chosen_idx]

def generate_text(prompt: str, model: dict, context_size: int, max_len: int = 30, temp: float = 1.0, top_k: int = 5):
    """接龍生成文字"""
    # 準備初始 context
    seed_chars = list(prompt)
    if len(seed_chars) >= context_size:
        context = seed_chars[-context_size:]
    else:
        # 如果 prompt 太短，用 <PAD> 補齊 (簡化處理)
        context = ["<PAD>"] * (context_size - len(seed_chars)) + seed_chars
        
    generated = list(prompt)
    
    for _ in range(max_len):
        current_context_tuple = tuple(context)
        next_char = sample_next(current_context_tuple, model, temp, top_k)
        
        if next_char == "<EOS>":
            break
            
        generated.append(next_char)
        # 滑動視窗：移除最舊的字，加入新字
        context = context[1:] + [next_char]
        
    return "".join(generated)

# ════════════════════════════════════════════════════════════
#  主程式與測試
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. 載入資料與建立模型
    corpus = load_corpus_from_url(CORPUS_URL)
    markov_model = build_markov_model(corpus, CONTEXT_SIZE)
    
    # 2. 測試生成
    test_prompts = ["台灣", "我們", "今天", "天氣", "小貓"]
    
    print("── 測試生成 ────────────────────────────")
    for prompt in test_prompts:
        result = generate_text(
            prompt=prompt, 
            model=markov_model, 
            context_size=CONTEXT_SIZE, 
            max_len=MAX_GEN_LEN,
            temp=TEMPERATURE,
            top_k=TOP_K
        )
        print(f"[{prompt}] → {result}")
    print("────────────────────────────────────────")
