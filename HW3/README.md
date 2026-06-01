# 深度學習作業：基於 nn0.py 的微型 Transformer 訓練範例

## 專案簡介
本專案為深度學習課程作業，基於輕量級深度學習框架 `nn0.py`，建構了一個微型的「下一個 Token 預測（Next-Token Prediction）」模型。本專案完整展示了如何利用底層的自動微分引擎，將各個神經網路組件串聯，並完成一次完整的訓練循環（Training Loop）。

## 檔案結構
* `nn0.py`: 底層深度學習框架，包含自動微分（Autograd）、Adam 優化器與 Transformer 相關算子。
* `HW3.py`: 主程式，負責初始化模型參數、定義前向傳播，並執行模型訓練。
* `README.md`: 本專案的說明文件。

## 執行環境與指令
本專案為純 Python 實作，僅依賴 Python 內建函式庫（如 `random`、`math`），**不需要**額外安裝 NumPy、PyTorch 等外部套件。

請在終端機（Terminal）中切換至本專案目錄，並執行以下指令：
```bash
python HW3.py
