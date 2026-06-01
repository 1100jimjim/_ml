# 深度學習作業：基於 nn0.py 的微型 Transformer 訓練範例

## 專案簡介
本專案為深度學習課程作業，基於輕量級深度學習框架 `nn0.py`，建構了一個微型的「下一個 Token 預測（Next-Token Prediction）」模型。本專案完整展示了如何利用底層的自動微分引擎，將各個神經網路組件串聯，並完成一次完整的訓練循環（Training Loop）。

## 檔案結構
* `nn0.py`: 底層深度學習框架，包含自動微分、Adam 優化器與 Transformer 相關算子。
* `train_example.py`: 主程式，負責初始化模型參數、定義前向傳播，並執行模型訓練。
* `README.md`: 本專案的說明文件。

## 執行環境與指令
本專案為純 Python 實作，僅依賴 Python 內建函式庫（如 `random`），**不需要**額外安裝 NumPy、PyTorch 等外部套件。

請在終端機（Terminal）中切換至本專案目錄，並執行以下指令：
python train_example.py

---

## 核心原理解析
本範例程式碼緊密結合了 `nn0.py` 的四大核心設計理念：

* **自動微分與動態計算圖 (`Value`)**
  模型中的所有權重矩陣（如 `W1`, `W2`）與輸入皆被封裝為 `Value` 物件。在前向傳播的運算過程中，系統利用運算子過載自動建構計算圖。當呼叫 `loss.backward()` 時，底層會執行拓撲排序與鏈式法則，自動求出所有參數的梯度。
* **Transformer 現代化算子**
  前向傳播（Forward Pass）中使用了 `linear` 進行矩陣乘法模擬全連接層，並串接了 `rmsnorm`。相較於傳統 LayerNorm，RMSNorm 移除了平移項，在提升計算效率的同時維持了模型訓練的數值穩定性。
* **損失函數與梯度清零**
  訓練過程採用了負對數似然（Negative Log-Likelihood）作為損失函數。在每次執行反向傳播前，嚴格呼叫 `optimizer.zero_grad()` 清空前一步的梯度，避免梯度錯誤累加。
* **Adam 優化演算法與學習率衰減**
  參數更新交由 `Adam` 優化器處理，利用其內建的一階（Momentum）與二階動量（Velocity）機制來平滑梯度。同時，在訓練迴圈中實作了學習率的線性衰減（Linear Decay），確保模型在逼近最佳解時能夠穩定收斂。

## 預期執行結果
執行 `train_example.py` 後，終端機會印出訓練過程的日誌（每 10 個 Epoch 輸出一次）。您將觀察到以下收斂現象：
* **Loss（損失值）**：隨著訓練步數增加，Loss 值會逐漸下降，最終逼近 0。
* **Target Token 機率**：模型對正確目標（Target ID）的預測機率會從初始的隨機分佈，穩定上升並逼近 1.0。
