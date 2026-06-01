import random
import math

# --- 1. 準備城市資料 ---
NUM_CITIES = 10
# 隨機產生 10 個城市的 (x, y) 座標
cities = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(1, NUM_CITIES + 1)}

# 計算兩城市之間的直線距離
def get_distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# --- 2. 定義解 (Solution) 的類別 ---
class TSPSolution:
    def __init__(self, route):
        self.route = route  # 儲存路徑，例如: [1, 2, 3, ..., 1]

    def str(self):
        # 將陣列轉換成 1=>2=>3 的字串格式
        return "=>".join(map(str, self.route))

    def height(self):
        # 提示 1：height 可以用旅行推銷員走的距離 * -1
        total_distance = 0
        for i in range(len(self.route) - 1):
            total_distance += get_distance(self.route[i], self.route[i+1])
        return -total_distance

    def neighbor(self):
        # 提示 1：選兩個邊 (a,b)(c,d) ，鄰居變成 (a,d), (b,c) (即 2-opt 交換)
        n = len(self.route) - 1  # 扣除最後重複回到起點的城市
        new_route = self.route[:-1] # 暫時去掉尾巴，方便處理

        # 隨機選擇兩個不相交的邊。i 為第一條邊的起點，j 為第二條邊的起點
        # 確保兩條邊不相鄰，否則反轉沒有意義
        i = random.randint(0, n - 3)
        j = random.randint(i + 2, n - 1)

        # 將 i+1 到 j 之間的路徑反轉，達到 (a,d)(b,c) 的連線效果
        new_route[i+1:j+1] = reversed(new_route[i+1:j+1])

        # 加回尾巴 (回到起點)
        new_route.append(new_route[0])
        
        return TSPSolution(new_route)


# --- 3. 爬山演算法主體 (根據您提供的參考資料) ---
def hillClimbing(s, maxGens, maxFails):
    print("start: ", s.str(), f"(距離: {-s.height():.2f})")
    fails = 0
    for gens in range(maxGens):
        snew = s.neighbor()
        sheight = s.height()
        nheight = snew.height()
        
        if (nheight >= sheight):          # 如果鄰近解比目前解更好 (距離更短，負值更大)
            print(f"gen {gens:4d} :", snew.str(), f"(距離: {-nheight:.2f})")
            s = snew                      # 就移動過去
            fails = 0                     # 移動成功，將連續失敗次數歸零
        else:
            fails = fails + 1             # 將連續失敗次數加一
            
        if (fails >= maxFails):
            print(f"連續失敗 {maxFails} 次，提早結束。")
            break
            
    print("solution: ", s.str(), f"(最終距離: {-s.height():.2f})")
    return s


# --- 4. 執行與測試 ---
if __name__ == "__main__":
    # 提示 2：初始解可以用 1=>2=>3=>.....=>n=>1
    initial_path = list(range(1, NUM_CITIES + 1))
    initial_path.append(1)  # 回到原點
    
    start_solution = TSPSolution(initial_path)
    
    # 執行爬山演算法：最多嘗試 10000 代，若連續 1000 次找不到更好的解就停下來
    hillClimbing(start_solution, 10000, 1000)
