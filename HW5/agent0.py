import os
import pathlib

class SecureFileManager:
    """安全檔案管理器：防止 Agent 越權存取目錄"""
    
    def __init__(self, base_dir="."):
        # 取得絕對路徑，作為安全邊界
        self.base_dir = pathlib.Path(base_dir).resolve()
        print(f"[系統初始化] Agent 工作目錄已鎖定於: {self.base_dir}")

    def _is_within_bounds(self, target_path):
        """檢查目標路徑是否在允許的資料夾範圍內 (防禦 ../ 穿越攻擊)"""
        resolved_target = pathlib.Path(target_path).resolve()
        try:
            # 嘗試計算相對路徑，若 target 不在 base 內會拋出 ValueError
            resolved_target.relative_to(self.base_dir)
            return True
        except ValueError:
            return False

    def _ask_for_permission(self, target_path, operation):
        """攔截並詢問使用者權限"""
        print(f"\n" + "="*50)
        print(f" ⚠️ [安全攔截] Agent 嘗試越權存取工作區外部檔案！")
        print(f" 動作: {operation}")
        print(f" 目標路徑: {target_path}")
        print(f" 工作目錄: {self.base_dir}")
        print("="*50)
        
        while True:
            choice = input("是否允許此存取操作？ (y/n): ").strip().lower()
            if choice == 'y':
                return True
            elif choice == 'n':
                return False
            print("請輸入 y 或 n。")

    def safe_open(self, file_path, mode='r', encoding='utf-8'):
        """安全開啟檔案的介面"""
        resolved_path = pathlib.Path(file_path).resolve()
        
        operation_name = "讀取" if 'r' in mode else "寫入"
        
        if self._is_within_bounds(resolved_path):
            # 在內部資料夾，直接放行
            return open(resolved_path, mode, encoding=encoding)
        else:
            # 在外部資料夾，攔截並詢問
            if self._ask_for_permission(resolved_path, operation_name):
                print(f"[系統提示] 已授權 {operation_name}: {resolved_path}\n")
                return open(resolved_path, mode, encoding=encoding)
            else:
                # 若拒絕，拋出權限錯誤
                raise PermissionError(f"系統或使用者拒絕存取外部檔案: {resolved_path}")


class BaseAgent:
    """基礎 Agent 類別，具備安全的檔案讀寫能力"""
    
    def __init__(self, name="Agent-007", work_dir="."):
        self.name = name
        self.file_manager = SecureFileManager(work_dir)

    def write_data(self, filepath, content):
        """Agent 寫入檔案的操作"""
        print(f"[{self.name}] 準備寫入檔案至: {filepath}")
        try:
            with self.file_manager.safe_open(filepath, 'w') as f:
                f.write(content)
            print(f"[{self.name}] ✅ 寫入成功！")
        except PermissionError as e:
            print(f"[{self.name}] ❌ 操作失敗: {e}")
        except Exception as e:
            print(f"[{self.name}] ❌ 發生未預期錯誤: {e}")

    def read_data(self, filepath):
        """Agent 讀取檔案的操作"""
        print(f"[{self.name}] 準備讀取檔案自: {filepath}")
        try:
            with self.file_manager.safe_open(filepath, 'r') as f:
                content = f.read()
            print(f"[{self.name}] ✅ 讀取成功！內容長度: {len(content)} 字元")
            return content
        except PermissionError as e:
            print(f"[{self.name}] ❌ 操作失敗: {e}")
        except FileNotFoundError:
            print(f"[{self.name}] ❌ 檔案不存在: {filepath}")
        except Exception as e:
            print(f"[{self.name}] ❌ 發生未預期錯誤: {e}")
        return None


# ==========================================
# 測試與執行區塊
# ==========================================
if __name__ == "__main__":
    # 建立一個測試用的工作資料夾
    WORK_DIR = "./agent_workspace"
    os.makedirs(WORK_DIR, exist_ok=True)
    
    # 實例化 Agent，並限制其活動範圍在 WORK_DIR
    agent = BaseAgent(name="Agent-Zero", work_dir=WORK_DIR)
    
    print("\n--- 測試 1: 合法的內部存取 ---")
    # Agent 嘗試寫入工作區內的檔案 (應該直接成功)
    internal_file = os.path.join(WORK_DIR, "internal_data.txt")
    agent.write_data(internal_file, "這是一份安全的內部資料。")
    agent.read_data(internal_file)

    print("\n--- 測試 2: 非法的外部存取 (路徑穿越攻擊) ---")
    # Agent 嘗試利用 ../ 寫入工作區外部的檔案 (應該觸發安全攔截)
    # 此路徑會被解析為與 agent_workspace 同級的目錄
    external_file = os.path.join(WORK_DIR, "../external_hacked_data.txt")
    
    print(f"\n[模擬 Agent 行為] 收到惡意指令，嘗試將資料外洩至: {external_file}")
    agent.write_data(external_file, "這是一份試圖寫入外部的未授權資料。")

    print("\n[系統提示] 測試結束。")
