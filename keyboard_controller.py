"""
键盘输入控制器

处理用户的键盘输入，支持实时目标更新、暂停/继续等功能
"""

import sys
import threading
import time
import queue
from typing import Tuple, Optional, Dict, Any
from gym_pybullet_drones.custom.config_continuous import INPUT_CONFIG, TESTING_SPACE

class KeyboardController:
    """键盘输入控制器，处理连续导航的用户交互"""
    
    def __init__(self):
        """初始化键盘控制器"""
        self.input_queue = queue.Queue()
        self.is_running = False
        self.input_thread = None
        self.current_command = None
        
        # 控制状态
        self.is_paused = False
        self.should_exit = False
        
        # 配置参数
        self.config = INPUT_CONFIG
        self.space_limits = TESTING_SPACE
        
        print(f"[键盘控制器] 已初始化")
        self._show_help()
    
    def start(self):
        """启动键盘输入监听"""
        if self.is_running:
            print(f"[键盘控制器] 已在运行中")
            return
            
        self.is_running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        print(f"[键盘控制器] 输入监听已启动")
        print(f"[键盘控制器] {self.config['prompt_message']}")
    
    def stop(self):
        """停止键盘输入监听"""
        self.is_running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        print(f"[键盘控制器] 输入监听已停止")
    
    def _input_loop(self):
        """输入监听循环（在独立线程中运行）"""
        while self.is_running:
            try:
                # 获取用户输入
                user_input = input().strip().lower()
                
                if user_input:
                    # 解析命令并放入队列
                    command = self._parse_command(user_input)
                    if command:
                        self.input_queue.put(command)
                        
                # 如果需要退出，则停止循环
                if self.should_exit:
                    break
                    
            except EOFError:
                # 处理输入结束
                break
            except KeyboardInterrupt:
                # 处理Ctrl+C
                self.input_queue.put({'type': 'exit', 'data': None})
                break
            except Exception as e:
                print(f"[键盘控制器] 输入错误: {e}")
    
    def _parse_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        解析用户输入命令
        
        参数:
            user_input: 用户输入的字符串
            
        返回:
            dict: 命令字典，包含type和data字段
        """
        # 移除多余空格并分割
        parts = user_input.split()
        
        if not parts:
            return None
            
        command = parts[0]
        
        # 处理不同类型的命令
        if command == 'exit' or command == 'quit':
            self.should_exit = True
            return {'type': 'exit', 'data': None}
            
        elif command == 'pause':
            self.is_paused = True
            return {'type': 'pause', 'data': None}
            
        elif command == 'resume':
            self.is_paused = False
            return {'type': 'resume', 'data': None}
            
        elif command == 'home':
            return {'type': 'home', 'data': None}
            
        elif command == 'current':
            return {'type': 'current', 'data': None}
            
        elif command == 'queue':
            return {'type': 'queue', 'data': None}
            
        elif command == 'clear':
            return {'type': 'clear', 'data': None}
            
        elif command == 'circle' or command == 'c':
            # 生成默认圆形轨迹（高度使用当前位置）
            return {'type': 'circle', 'data': {'radius': 2.0, 'height': None, 'waypoints': 100, 'clockwise': False}}
            
        elif command == 'circle_cw' or command == 'cc':
            # 生成顺时针圆形轨迹（高度使用当前位置）
            return {'type': 'circle', 'data': {'radius': 2.0, 'height': None, 'waypoints': 100, 'clockwise': True}}
            
        elif command == 'circle_avoid' or command == 'ca':
            # 生成带大量障碍的圆形避障任务（用于测试避障能力）
            # 默认: radius=1.0, waypoints=64, num_obstacles=16, obs_radius=0.12, height=None
            return {'type': 'circle_avoid', 'data': {'radius': 1.0, 'waypoints': 64, 'num_obstacles': 16, 'obs_radius': 0.12, 'height': None}}
            
        elif command == 'stats' or command == 's':
            # 显示轨迹统计
            return {'type': 'stats', 'data': None}
            
        elif command == 'visual' or command == 'v':
            # 显示轨迹可视化
            return {'type': 'visual', 'data': None}
            
        elif command == 'help':
            self._show_help()
            return None
            
        elif len(parts) == 3:
            # 尝试解析坐标 (x, y, z)
            try:
                x = float(parts[0])
                y = float(parts[1]) 
                z = float(parts[2])
                
                # 验证坐标是否在安全范围内
                if self._validate_coordinates(x, y, z):
                    target_pos = [x, y, z]
                    return {'type': 'target', 'data': target_pos}
                else:
                    return None
                    
            except ValueError:
                print(f"[错误] 无效的坐标格式，请输入数字: {user_input}")
                return None
                
        else:
            print(f"[错误] 无法识别的命令: {user_input}")
            print(f"[提示] 输入 'help' 查看帮助信息")
            return None
    
    def _validate_coordinates(self, x: float, y: float, z: float) -> bool:
        """
        验证坐标是否在安全范围内
        
        参数:
            x, y, z: 目标坐标
            
        返回:
            bool: 坐标是否有效
        """
        x_min, x_max = self.space_limits['x_range']
        y_min, y_max = self.space_limits['y_range']
        z_min, z_max = self.space_limits['z_range']
        
        # 添加安全边界
        safety_margin = 0.3
        
        if not (x_min + safety_margin <= x <= x_max - safety_margin):
            print(f"[错误] X坐标超出范围 [{x_min + safety_margin:.1f}, {x_max - safety_margin:.1f}]: {x}")
            return False
            
        if not (y_min + safety_margin <= y <= y_max - safety_margin):
            print(f"[错误] Y坐标超出范围 [{y_min + safety_margin:.1f}, {y_max - safety_margin:.1f}]: {y}")
            return False
            
        if not (z_min + safety_margin <= z <= z_max - safety_margin):
            print(f"[错误] Z坐标超出范围 [{z_min + safety_margin:.1f}, {z_max - safety_margin:.1f}]: {z}")
            return False
            
        return True
    
    def _show_help(self):
        """显示帮助信息"""
        print(self.config['help_message'])
        
        # 显示当前空间范围
        x_range = self.space_limits['x_range']
        y_range = self.space_limits['y_range'] 
        z_range = self.space_limits['z_range']
        
        print(f"当前飞行空间范围:")
        print(f"  X轴: {x_range[0]} 到 {x_range[1]} 米")
        print(f"  Y轴: {y_range[0]} 到 {y_range[1]} 米") 
        print(f"  Z轴: {z_range[0]} 到 {z_range[1]} 米")
        print()
    
    def get_command(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的用户命令
        
        返回:
            dict: 命令字典，如果没有新命令则返回None
        """
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None
    
    def has_commands(self) -> bool:
        """
        检查是否有待处理的命令
        
        返回:
            bool: 是否有命令等待处理
        """
        return not self.input_queue.empty()
    
    def clear_commands(self):
        """清空命令队列"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
                
        print(f"[键盘控制器] 命令队列已清空")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取控制器状态
        
        返回:
            dict: 状态信息
        """
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'should_exit': self.should_exit,
            'pending_commands': self.input_queue.qsize(),
        }
    
    def set_pause(self, paused: bool):
        """
        设置暂停状态
        
        参数:
            paused: 是否暂停
        """
        self.is_paused = paused
        if paused:
            print(f"[控制] 无人机导航已暂停")
        else:
            print(f"[控制] 无人机导航已恢复")


class StatusDisplayer:
    """状态显示器，用于实时显示无人机状态"""
    
    def __init__(self, update_frequency=2.0):
        """
        初始化状态显示器
        
        参数:
            update_frequency: 更新频率（秒）
        """
        self.update_frequency = update_frequency
        self.last_update_time = 0
        self.display_enabled = True
    
    def update_display(self, drone_state: Dict[str, Any], controller_status: Dict[str, Any]):
        """
        更新状态显示
        
        参数:
            drone_state: 无人机状态信息
            controller_status: 控制器状态信息
        """
        current_time = time.time()
        
        # 检查是否需要更新显示
        if current_time - self.last_update_time < self.update_frequency:
            return
        
        if not self.display_enabled:
            return
            
        self.last_update_time = current_time
        
        # 清屏并显示状态（简单版本，避免屏幕闪烁）
        print("\r" + "="*60, end='')
        print(f"\r[状态] 位置: ({drone_state['position'][0]:.2f}, {drone_state['position'][1]:.2f}, {drone_state['position'][2]:.2f}) | "
              f"目标: ({drone_state['target_position'][0]:.2f}, {drone_state['target_position'][1]:.2f}, {drone_state['target_position'][2]:.2f}) | "
              f"距离: {drone_state['distance_to_target']:.2f}m | "
              f"{'暂停' if controller_status['is_paused'] else '导航中'}", end='', flush=True)
    
    def show_status_summary(self, drone_state: Dict[str, Any]):
        """
        显示详细状态摘要
        
        参数:
            drone_state: 无人机状态信息
        """
        print(f"\n" + "="*60)
        print(f"无人机状态摘要:")
        print(f"  当前位置: ({drone_state['position'][0]:.3f}, {drone_state['position'][1]:.3f}, {drone_state['position'][2]:.3f})")
        print(f"  当前速度: ({drone_state['velocity'][0]:.3f}, {drone_state['velocity'][1]:.3f}, {drone_state['velocity'][2]:.3f})")
        print(f"  目标位置: ({drone_state['target_position'][0]:.3f}, {drone_state['target_position'][1]:.3f}, {drone_state['target_position'][2]:.3f})")
        print(f"  到目标距离: {drone_state['distance_to_target']:.3f} 米")
        print(f"  飞行时间: {drone_state['time_elapsed']:.1f} 秒")
        print(f"  {'✅ 已到达目标' if drone_state['is_near_target'] else '🚁 正在飞行'}")
        print("="*60)
    
    def enable_display(self, enabled: bool):
        """启用或禁用状态显示"""
        self.display_enabled = enabled
