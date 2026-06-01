# MicroGPT: 本地端 AI 輔助開發實作專案

## 專案簡介
本專案為機器學習課程作業，目標為實作一個微型 GPT (Generative Pre-trained Transformer) 模型架構。
為符合題目要求，本專案全程採用**本地端 AI 開發代理工具 (Aider + 本地開源模型)** 進行輔助開發與自動化除錯（Agentic Coding），並成功完成了模型建構與初步的訓練驗證。

##  開發環境與工具
* **程式語言:** Python 3.10+
* **核心框架:** PyTorch
* **AI 開發工具:** Aider (搭配本地 Ollama 模型)
* **開發環境:** VS Code + Windows PowerShell

##  模型架構
本專案手刻了標準的 GPT 底層神經網路架構，包含以下核心模組：
1. **Token Embedding:** 詞彙嵌入層 (`nn.Embedding`)。
2. **Causal Self-Attention (因果自注意力機制):** * 實作了 Query, Key, Value 的線性投影。
   * 包含 Masked Attention 的概念，利用矩陣相乘 (`q @ k.transpose`) 與 Softmax 計算注意力權重。
3. **Linear Projection:** 輸出層，將注意力機制的結果映射回詞彙表大小 (`vocab_size`) 以計算機率。

##  人機協作與開發歷程 (AI 協作亮點)
在開發過程中，我扮演「架構總監」的角色，引導 AI 代理撰寫程式碼並解決深層 Bug：
1. **架構重構 (Refactoring):** 導正 AI 過度複雜的類別繼承，強制其使用標準的 `forward` 函式傳遞張量 (Tensor)，解決了底層 `AttributeError` 的環境衝突。
2. **維度對齊 (Shape Debugging):** 成功引導 AI 解決了在矩陣相乘時最棘手的 `RuntimeError: mat1 and mat2 shapes cannot be multiplied` 維度不匹配問題。
3. **損失函數修正:** 修正了生成測試資料時 `outputs` 與 `targets` 批次大小 (Batch Size) 不一致的問題，確保 `CrossEntropyLoss` 能正確運作。

##  執行方式與結果
**執行指令:**
```bash
python microgpt_auto.py
