# 🧠 Neural Network Engine: From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Required-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange.svg)
![License](https://img.shields.io/badge/License-MIT-success.svg)

本專案為機器學習相關課程之期末專案。旨在**「不依賴現成深度學習框架（如 PyTorch、TensorFlow）」**的前提下，純粹運用 Python 與 NumPy 從零開始（From Scratch）建構一個輕量級神經網路引擎，並將其應用於**體育賽事量化預測**之實戰情境。

## ✨ 核心特色 (Key Features)

* **純粹的數學實作**：底層全連接層 (Dense Layer) 的矩陣相乘與空間轉換。
* **非線性處理**：實作 ReLU 激勵函數以過濾負值訊號，並運用 Softmax 函數輸出機率分佈。
* **自研優化器與反向傳播**：實作分類交叉熵 (Categorical Cross-Entropy) 損失函數，並結合 Softmax 推導出極簡的微積分連鎖律 (Chain Rule) 梯度公式，最後透過 SGD (隨機梯度下降) 更新權重。
* **量化賽事預測應用**：設計專屬的資料處理管線，透過預期進球 (xG)、戰術優勢等 5 項量化指標，精準預測賽事勝負。

---

## 📂 專案架構 (Project Structure)

```text
📦 HW-ML
 ┣ 📜 nn_engine.py      # 神經網路引擎核心程式碼與訓練迴圈
 ┗ 📜 README.md         # 專案說明文件
