#!/usr/bin/env python3
"""
独立键盘控制器
在单独的终端中运行，用于控制连续导航系统
避免输出信息刷掉键盘输入
"""

import socket
import time
import json
from gym_pybullet_drones.custom.config_continuous import INPUT_CONFIG, RECOMMENDED_TARGETS

class RemoteController:
    """独立的键盘控制器，通过网络发送命令到导航系统"""
    
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
    def connect(self):
        """连接到导航系统"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✅ 已连接到导航系统 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            print(f"   请确保导航系统正在运行")
            return False
    
    def send_command(self, command_type, data=None):
        """发送命令到导航系统"""
        try:
            message = {
                'type': command_type,
                'data': data,
                'timestamp': time.time()
            }
            message_str = json.dumps(message) + '\n'
            self.socket.send(message_str.encode())
            return True
        except Exception as e:
            print(f"❌ 发送命令失败: {e}")
            return False
    
    def show_help(self):
        """显示帮助信息"""
        print("""
🎮 独立键盘控制器 - 连续导航系统
================================================

基础命令:
  x y z        - 设置新目标点 (例: 0.8 0.5 1.2)
  pause        - 暂停无人机
  resume       - 继续导航
  home         - 返回起始位置
  current      - 显示当前位置和队列状态
  queue        - 显示目标队列
  clear        - 清空目标队列
  help         - 显示帮助信息
  exit         - 退出控制器

🤖 LLM智能轨迹命令:
  circle       - 生成逆时针圆形飞行轨迹 (半径2.5m)
  circle_cw    - 生成顺时针圆形飞行轨迹 (半径2.5m)
  big_circle   - 生成大圆形轨迹 (半径3.2m)
  small_circle - 生成小圆形轨迹 (半径1.8m)
  circle_avoid - 生成带障碍的圆形避障任务（快捷: ca）
  stats        - 显示轨迹统计信息
  visual       - 显示轨迹可视化图表

快速测试命令:
  test1        - 设置推荐目标1: [0.5, 0.5, 1.2]
  test2        - 设置推荐目标2: [0.8, 0.3, 1.1] 
  test3        - 设置推荐目标3: [0.3, 0.8, 1.3]
  test4        - 设置推荐目标4: [0.0, 0.5, 1.4]
  test5        - 设置推荐目标5: [-0.5, 0.0, 1.2]
  testall      - 依次添加所有测试目标

连续导航说明:
- 无人机按顺序访问目标点 (a→b→c)
- 到达当前目标后自动前往下一个
- 可随时添加新目标到队列末尾

================================================
        """)
    
    def parse_command(self, user_input):
        """解析用户输入命令"""
        parts = user_input.strip().split()
        if not parts:
            return None
            
        command = parts[0].lower()
        
        # 快速测试命令
        if command.startswith('test'):
            if command == 'testall':
                print("📋 添加所有测试目标到队列...")
                for i, target in enumerate(RECOMMENDED_TARGETS):
                    self.send_command('target', target)
                    print(f"   ✅ 目标{i+1}: {target}")
                return None
            elif command in ['test1', 'test2', 'test3', 'test4', 'test5']:
                idx = int(command[4]) - 1
                target = RECOMMENDED_TARGETS[idx]
                print(f"🎯 设置测试目标{idx+1}: {target}")
                return {'type': 'target', 'data': target}
        
        # LLM圆形轨迹命令 - 适应训练空间[-1.5, 1.5]
        if command == 'circle':
            print("🔄 生成逆时针圆形轨迹 (半径0.8m, 12个轨迹点)...")
            return {'type': 'circle', 'data': {'radius': 0.8, 'clockwise': False, 'waypoints': 48}}
        elif command == 'circle_cw':
            print("🔄 生成顺时针圆形轨迹 (半径0.8m, 12个轨迹点)...")
            return {'type': 'circle', 'data': {'radius': 0.8, 'clockwise': True, 'waypoints': 48}}
        elif command == 'big_circle':
            print("🔄 生成大圆形轨迹 (半径1.0m, 16个轨迹点)...")
            return {'type': 'circle', 'data': {'radius': 1.0, 'clockwise': False, 'waypoints': 64}}
        elif command == 'small_circle':
            print("🔄 生成小圆形轨迹 (半径0.6m, 8个轨迹点)...")
            return {'type': 'circle', 'data': {'radius': 0.6, 'clockwise': False, 'waypoints': 32}}
        elif command == 'circle_avoid' or command == 'ca':
            print("🔄 生成带障碍的圆形避障任务 (默认: radius=1.0m, 16个障碍)...")
            return {'type': 'circle_avoid', 'data': {'radius': 1.0, 'waypoints': 64, 'num_obstacles': 16, 'obs_radius': 0.12, 'height': None}}
        elif command == 'stats':
            print("📊 显示轨迹统计信息...")
            return {'type': 'stats', 'data': None}
        elif command == 'visual':
            print("📊 显示轨迹可视化...")
            return {'type': 'visual', 'data': None}
        
        # 基础命令
        if command in ['pause', 'resume', 'home', 'current', 'queue', 'clear']:
            return {'type': command, 'data': None}
        elif command in ['exit', 'quit']:
            return {'type': 'exit', 'data': None}
        elif command == 'help':
            self.show_help()
            return None
        elif len(parts) == 3:
            # 坐标命令
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                # 检查坐标范围
                if abs(x) > 2.0 or abs(y) > 2.0 or z < 0.5 or z > 2.5:
                    print(f"⚠️  建议坐标范围: X/Y[-2.0, 2.0], Z[0.5, 2.5]")
                    print(f"   当前输入: ({x:.2f}, {y:.2f}, {z:.2f})")
                
                return {'type': 'target', 'data': [x, y, z]}
            except ValueError:
                print(f"❌ 坐标格式错误: {user_input}")
                return None
        else:
            print(f"❌ 未知命令: {command}")
            print("   输入 'help' 查看帮助信息")
            return None
    
    def run(self):
        """运行控制器主循环"""
        print("🚁 无人机独立键盘控制器")
        print("=" * 50)
        
        if not self.connect():
            return
        
        self.show_help()
        self.running = True
        
        try:
            while self.running:
                try:
                    user_input = input("\n🎮 输入命令: ").strip()
                    
                    if not user_input:
                        continue
                    
                    command = self.parse_command(user_input)
                    if command is None:
                        continue
                    
                    if command['type'] == 'exit':
                        print("👋 退出控制器...")
                        break
                    
                    success = self.send_command(command['type'], command['data'])
                    if success:
                        print(f"✅ 命令已发送: {command['type']}")
                    
                except KeyboardInterrupt:
                    print("\n👋 用户中断，退出控制器...")
                    break
                except Exception as e:
                    print(f"❌ 输入处理错误: {e}")
                    
        finally:
            if self.socket:
                self.socket.close()
            print("🔌 连接已关闭")

def main():
    """主函数"""
    controller = RemoteController()
    controller.run()

if __name__ == "__main__":
    main()
