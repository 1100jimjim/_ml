# 🧠 Neural Network Engine: From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Required-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange.svg)
![Machine Learning](https://img.shields.io/badge/Domain-Sports_Quant-red.svg)
![License](https://img.shields.io/badge/License-MIT-success.svg)

本專案為機器學習從零開發 (From Scratch) 的核心實作。旨在**不依賴任何現成深度學習框架（如 PyTorch、TensorFlow）**的前提下，純粹運用 Python 與 NumPy 建構輕量級神經網路引擎，並針對「體育賽事量化預測」構建端到端 (End-to-End) 的數據分析管線。

---

## 專案核心價值 (Core Value Proposition)

1. **白盒化底層引擎 (White-box Engine)**：
   完全掌控神經網路的數學底層，從全連接層的空間映射、ReLU 非線性轉換，到 Softmax 與 Cross-Entropy 結合的微積分連鎖律 (Chain Rule) 梯度推導，皆以高可讀性之 Python 程式碼實作。
2. **數據導向量化預測 (Data-Driven Forecasting)**：
   摒棄直覺猜測，專為賽事分析設計的輸入層，能有效處理複雜的非線性特徵。
3. **高防呆與邊界保護 (Edge-case Handling)**：
   引擎內建數值溢位防護 (如 Softmax 最大值平移) 與 `log(0)` 裁切保護，確保在極端輸入值下系統依然穩定運行。

---

## 系統架構與特徵工程 (Architecture & Feature Engineering)

本模型採用 `5 -> 16 -> 3` 的網路拓樸架構。為了讓引擎能精準捕捉賽事規律，輸入層特別設計了 **5 階段量化評估框架**：

* **Input 1: 預期進球值 (xG) 差異** - 衡量雙方創造絕對得分機會的質量。
* **Input 2: 傷兵與陣容完整度 (Injury Impact)** - 量化核心球員缺陣對整體戰力的折損。
* **Input 3: 戰術克制指數 (Tactical Matchup)** - 評估雙方陣型與攻守節奏的相剋關係。
* **Input 4: 主隊疲勞與主場優勢** - 結合賽程密集度與主場勝率加權。
* **Input 5: 客隊疲勞指數** - 客場長途跋涉與休息天數的負向權重。

隱藏層採用 **ReLU** 函數過濾無效噪訊，輸出層則透過 **Softmax** 將運算結果轉化為 3 種賽果（主勝、平局、客勝）的**絕對機率分佈**。

---

## 數學推導與效能表現 (Mathematics & Performance)

### 極簡化反向傳播 (Optimized Backpropagation)
本專案在效能上的最大亮點，在於數學層面的化簡。將 Softmax 激勵函數與分類交叉熵 (Categorical Cross-Entropy) 損失函數合併求導，將原本複雜的雅可比矩陣 (Jacobian matrix) 運算，化簡為優雅的：

`dZ = Predicted_Probability - True_Label`

這項優化大幅降低了 CPU 的運算開銷，使得 SGD (隨機梯度下降) 優化器能以極高的效率更新權重。

### 實驗結果 (Training Results)
在 1000 筆模擬賽事數據、學習率 (LR) 設為 0.8 的環境下進行 100 Epochs 訓練：
* **Loss 收斂**：從 `1.09` 迅速且平滑地下降至 **`0.09`**。
* **Accuracy 提升**：從隨機瞎猜的 `26%` 攀升至 **`99.5%`**。
*(註：執行 `python nn_engine.py` 將自動產出具備雙 Y 軸的效能收斂 Matplotlib 圖表)*

---

## 未來應用藍圖 (Roadmap & Future Work)

本引擎的最終目標不僅僅是預測勝率，而是打造一套完整的**體育量化投資系統**：

1. **真實聯賽 API 整合 (Data Pipeline)**
   未來將直接串接英超 (Premier League)、德甲 (Bundesliga)、義甲 (Serie A) 等五大聯賽的即時數據庫，自動餵入 xG 與傷兵報告進行訓練。
2. **結合 Kelly Criterion (凱利公式計算器)**
   將神經網路 Softmax 輸出的「預測機率」與台灣運彩等平台提供的「賠率」自動對接。透過 Python 實作的凱利公式，在面對 $t=0$ 或負期望值時自動給出「不建議下注」的防呆提示，從而實現動態的最佳化資金分配 (Position Sizing)。
3. **演算法升級 (Algorithm Enhancement)**
   引入 Adam 優化器以處理更稀疏的特徵，並加入 Dropout 層以防止模型在單一球隊數據上過度擬合 (Overfitting)。

---
