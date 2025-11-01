#!/usr/bin/env python3
"""
SAC 连续导航系统 - 独立远程控制器

在单独的终端中运行，通过网络控制 SAC 导航系统
避免导航系统的日志输出干扰键盘输入
"""

import socket
import time
import json
import sys
from typing import Optional, Dict, Any

# 默认配置
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8888  # SAC 导航系统使用的网络端口

# 推荐的测试目标点（在训练空间内）
RECOMMENDED_TARGETS = [
    [0.5, 0.5, 0.3],   # 测试点 1
    [0.8, 0.3, 0.3],   # 测试点 2
    [0.3, 0.8, 0.3],   # 测试点 3
    [-0.5, 0.5, 0.3],  # 测试点 4
    [-0.5, -0.5, 0.3], # 测试点 5
]


class SACRemoteController:
    """SAC 连续导航系统的独立远程控制器"""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        初始化远程控制器
        
        参数:
            host: 导航系统主机地址
            port: 导航系统监听端口
        """
        self.host = host
        self.port = port
        self.running = False
        
    def send_command(self, command_type: str, target: Optional[list] = None) -> bool:
        """
        发送命令到导航系统
        
        参数:
            command_type: 命令类型 ('target', 'home', 等)
            target: 目标坐标 [x, y, z] (仅用于 'target' 命令)
            
        返回:
            bool: 是否发送成功
        """
        try:
            # 创建新的连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.host, self.port))
            
            # 构建命令
            command = {'type': command_type}
            if target is not None:
                command['target'] = target
            
            # 发送命令
            message = json.dumps(command)
            sock.send(message.encode('utf-8'))
            
            # 接收响应
            response = sock.recv(1024).decode('utf-8')
            result = json.loads(response)
            
            sock.close()
            
            if result.get('status') == 'success':
                print(f"✅ {result.get('message', '命令执行成功')}")
                return True
            else:
                print(f"❌ {result.get('message', '命令执行失败')}")
                return False
                
        except ConnectionRefusedError:
            print(f"❌ 无法连接到导航系统 {self.host}:{self.port}")
            print("   请确保 start_sac_continuous.py 正在运行")
            return False
        except socket.timeout:
            print("❌ 连接超时")
            return False
        except Exception as e:
            print(f"❌ 发送命令失败: {e}")
            return False
    
    def show_help(self):
        """显示帮助信息"""
        print("""
╔════════════════════════════════════════════════════════════╗
║       🚁 SAC 连续导航系统 - 远程控制器                    ║
╚════════════════════════════════════════════════════════════╝

📍 目标控制命令:
  x y z          设置新目标点 (例: 0.5 0.5 0.3)
  home           返回起始位置 [0, 0, 0.1]
  
🎯 快速测试命令:
  test1          前往测试点 1: [0.5, 0.5, 0.3]
  test2          前往测试点 2: [0.8, 0.3, 0.3]
  test3          前往测试点 3: [0.3, 0.8, 0.3]
  test4          前往测试点 4: [-0.5, 0.5, 0.3]
  test5          前往测试点 5: [-0.5, -0.5, 0.3]
  testall        依次访问所有测试点
  
📦 预定义任务:
  square         正方形巡航
  triangle       三角形巡航
  zigzag         Z字形巡航
  spiral         螺旋上升
  
ℹ️  其他命令:
  help           显示此帮助信息
  status         显示连接状态
  exit           退出控制器
  
💡 提示:
  - 训练空间范围: X/Y ∈ [-1.5, 1.5], Z ∈ [0.05, 2.5]
  - 建议测试范围: X/Y ∈ [-0.8, 0.8], Z ∈ [0.2, 0.5]
  - 使用较低高度 (0.3m) 以提高成功率
  - 每个命令发送一个目标，无人机按顺序导航
  
══════════════════════════════════════════════════════════════
        """)
    
    def parse_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        解析用户输入命令
        
        参数:
            user_input: 用户输入的字符串
            
        返回:
            Dict: 包含 type 和可选 target 的命令字典，或 None
        """
        parts = user_input.strip().split()
        if not parts:
            return None
        
        command = parts[0].lower()
        
        # 快速测试命令
        if command == 'testall':
            print("📋 发送所有测试点到队列...")
            for i, target in enumerate(RECOMMENDED_TARGETS, 1):
                if self.send_command('target', target):
                    print(f"   ✓ 测试点 {i}: {target}")
                time.sleep(0.2)  # 避免发送过快
            return None
        
        elif command.startswith('test') and len(command) == 5:
            try:
                idx = int(command[4]) - 1
                if 0 <= idx < len(RECOMMENDED_TARGETS):
                    target = RECOMMENDED_TARGETS[idx]
                    print(f"🎯 设置测试点 {idx+1}: {target}")
                    return {'type': 'target', 'target': target}
                else:
                    print(f"❌ 测试点索引超出范围 (1-{len(RECOMMENDED_TARGETS)})")
                    return None
            except (ValueError, IndexError):
                print(f"❌ 无效的测试命令: {command}")
                return None
        
        # 预定义任务
        elif command == 'square':
            print("🔲 正方形巡航任务...")
            targets = [
                [0.5, 0.5, 0.3],
                [0.5, -0.5, 0.3],
                [-0.5, -0.5, 0.3],
                [-0.5, 0.5, 0.3],
                [0.0, 0.0, 0.3],  # 返回中心
            ]
            for i, target in enumerate(targets, 1):
                if self.send_command('target', target):
                    print(f"   ✓ 航点 {i}/5: {target}")
                time.sleep(0.2)
            return None
        
        elif command == 'triangle':
            print("🔺 三角形巡航任务...")
            targets = [
                [0.6, 0.0, 0.3],
                [-0.3, 0.5, 0.3],
                [-0.3, -0.5, 0.3],
                [0.0, 0.0, 0.3],  # 返回中心
            ]
            for i, target in enumerate(targets, 1):
                if self.send_command('target', target):
                    print(f"   ✓ 航点 {i}/4: {target}")
                time.sleep(0.2)
            return None
        
        elif command == 'zigzag':
            print("⚡ Z字形巡航任务...")
            targets = [
                [0.7, 0.5, 0.3],
                [-0.7, 0.5, 0.3],
                [0.7, 0.0, 0.3],
                [-0.7, 0.0, 0.3],
                [0.7, -0.5, 0.3],
                [-0.7, -0.5, 0.3],
                [0.0, 0.0, 0.3],  # 返回中心
            ]
            for i, target in enumerate(targets, 1):
                if self.send_command('target', target):
                    print(f"   ✓ 航点 {i}/7: {target}")
                time.sleep(0.2)
            return None
        
        elif command == 'spiral':
            print("🌀 螺旋上升任务...")
            import math
            targets = []
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                radius = 0.5
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 0.2 + i * 0.02  # 逐步上升
                targets.append([round(x, 2), round(y, 2), round(z, 2)])
            
            for i, target in enumerate(targets, 1):
                if self.send_command('target', target):
                    print(f"   ✓ 航点 {i}/8: {target}")
                time.sleep(0.2)
            return None
        
        # 基础命令
        elif command == 'home':
            print("🏠 返回起点...")
            return {'type': 'home'}
        
        elif command == 'help':
            self.show_help()
            return None
        
        elif command == 'status':
            print(f"📡 连接状态: {self.host}:{self.port}")
            # 尝试连接测试
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((self.host, self.port))
                sock.close()
                print("   ✅ 导航系统在线")
            except:
                print("   ❌ 无法连接到导航系统")
            return None
        
        elif command in ['exit', 'quit', 'q']:
            return {'type': 'exit'}
        
        # 坐标命令 (x y z)
        elif len(parts) == 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                
                # 检查坐标范围并给出警告
                if abs(x) > 1.5 or abs(y) > 1.5:
                    print(f"⚠️  警告: X/Y 超出训练范围 [-1.5, 1.5]")
                    print(f"   当前: X={x:.2f}, Y={y:.2f}")
                    confirm = input("   是否继续? (y/n): ").lower()
                    if confirm != 'y':
                        print("   ❌ 已取消")
                        return None
                
                if z < 0.05 or z > 2.5:
                    print(f"⚠️  警告: Z 超出范围 [0.05, 2.5]")
                    print(f"   当前: Z={z:.2f}")
                    confirm = input("   是否继续? (y/n): ").lower()
                    if confirm != 'y':
                        print("   ❌ 已取消")
                        return None
                
                return {'type': 'target', 'target': [x, y, z]}
            
            except ValueError:
                print(f"❌ 坐标格式错误: {user_input}")
                print("   格式: x y z  (例: 0.5 0.5 0.3)")
                return None
        
        else:
            print(f"❌ 未知命令: {command}")
            print("   输入 'help' 查看帮助")
            return None
    
    def run(self):
        """运行控制器主循环"""
        print("\n" + "="*60)
        print("  🚁 SAC 连续导航系统 - 远程控制器")
        print("="*60)
        print(f"  目标系统: {self.host}:{self.port}")
        print("  输入 'help' 查看命令列表")
        print("="*60 + "\n")
        
        self.running = True
        
        try:
            while self.running:
                try:
                    user_input = input("🎮 SAC> ").strip()
                    
                    if not user_input:
                        continue
                    
                    command = self.parse_command(user_input)
                    
                    if command is None:
                        continue
                    
                    if command['type'] == 'exit':
                        print("👋 退出控制器...")
                        break
                    
                    # 发送命令
                    self.send_command(
                        command['type'],
                        command.get('target')
                    )
                
                except KeyboardInterrupt:
                    print("\n\n👋 用户中断，退出控制器...")
                    break
                except EOFError:
                    print("\n👋 输入结束，退出控制器...")
                    break
                except Exception as e:
                    print(f"❌ 错误: {e}")
        
        finally:
            self.running = False
            print("🔌 控制器已关闭\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SAC 连续导航系统 - 远程控制器'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=DEFAULT_HOST,
        help=f'导航系统主机地址 (默认: {DEFAULT_HOST})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help=f'导航系统端口 (默认: {DEFAULT_PORT})'
    )
    
    args = parser.parse_args()
    
    # 创建并运行控制器
    controller = SACRemoteController(
        host=args.host,
        port=args.port
    )
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n👋 程序被中断")
        sys.exit(0)


if __name__ == "__main__":
    main()
