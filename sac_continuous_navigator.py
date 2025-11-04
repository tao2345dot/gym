"""
SAC 连续导航主控制器

整合环境、键盘输入、SAC模型推理、LLM避障等功能,实现连续导航系统
支持多障碍物环境下的连续目标导航
"""

# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalSubscript=false  
# pyright: reportGeneralTypeIssues=false

import os
import sys
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass

import time
import socket
import json
import threading
import numpy as np
import pybullet as p
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from stable_baselines3 import SAC

from gym_pybullet_drones.custom.space_expander import ExtendedHoverAviary
from gym_pybullet_drones.custom.keyboard_controller import KeyboardController, StatusDisplayer
from gym_pybullet_drones.custom.config_continuous import *

# 导入避障规划器（可选）
try:
    from gym_pybullet_drones.custom.llm_obstacle_avoidance import plan_avoidance
    AVOIDANCE_AVAILABLE = True
    print("[避障模块] ✅ LLM避障规划器加载成功")
except Exception:
    AVOIDANCE_AVAILABLE = False
    print("[避障模块] ⚠️ 未找到避障规划器，使用内置简单避障策略")

# 条件导入，避免未绑定错误
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from gym_pybullet_drones.custom.llm_circle_planner import generate_circle_trajectory
try:
    from gym_pybullet_drones.custom.llm_circle_planner import generate_circle_trajectory
    LLM_AVAILABLE = True
    print("[LLM模块] ✅ LLM轨迹规划器加载成功")
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"[LLM模块] ⚠️ LLM轨迹规划器未加载: {e}")

# 导入可视化工具
try:
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    VISUALIZATION_AVAILABLE = True
    # 配置中文字体
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['axes.unicode_minus'] = False
    print("[可视化] ✅ 轨迹可视化功能可用")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[可视化] ⚠️ matplotlib未安装，轨迹可视化不可用")

# 柱子固定配置
PILLAR_CONFIG = {
    'radius': 0.08,      # 柱子半径（米） -> 调整为与蓝色路径点大小一致
    'height': 0.5,       # 柱子高度（米） -> 2.0 减去 3/4 = 0.5
    'count': 4,          # 柱子数量
    'color': [0.6, 0.6, 0.6, 1.0],  # 灰色柱子
}

# 网络控制配置
NETWORK_CONFIG = {
    'host': 'localhost',
    'port': 8888,
    'timeout': 1.0,
}

# 渲染配置
RENDER_CONFIG = {
    'width': 1280,
    'height': 720,
    'fps': 30,
}

# 生成随机柱子位置的函数
def generate_random_pillars(count=2, radius=0.08, height=2.0, 
                           x_range=(-1.5, 1.5), y_range=(-1.5, 1.5),
                           min_distance=0.8, origin_safe_radius=0.5):
    """
    生成随机位置的柱子
    
    参数:
        count: 柱子数量
        radius: 柱子半径
        height: 柱子高度
        x_range: X 坐标范围
        y_range: Y 坐标范围
        min_distance: 柱子之间最小距离
        origin_safe_radius: 原点附近的安全半径（避免柱子生成在起点）
    
    返回:
        List[dict]: 柱子配置列表
    """
    pillars = []
    max_attempts = 100
    
    for i in range(count):
        attempts = 0
        while attempts < max_attempts:
            # 随机生成位置
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = height / 2  # Z 坐标为柱子高度的一半（PyBullet 圆柱体中心点）
            
            pos = np.array([x, y, z])
            
            # 检查是否在原点安全半径内
            if np.linalg.norm(pos[:2]) < origin_safe_radius:
                attempts += 1
                continue
            
            # 检查与已有柱子的距离
            valid = True
            for existing in pillars:
                existing_pos = np.array(existing['pos'])
                distance = np.linalg.norm(pos[:2] - existing_pos[:2])
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                pillars.append({
                    'pos': [x, y, z],
                    'radius': radius,
                    'height': height
                })
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"[警告] 柱子 {i+1} 生成失败，已尝试 {max_attempts} 次")
    
    return pillars

# 初始化障碍物列表（启动时会重新生成）
OBSTACLES = []

# 避障参数
COLLISION_THRESHOLD = 0.15  # 碰撞检测距离阈值（障碍物半径 + 安全余量）- 从0.5降低到0.15
WAYPOINT_REACH_THRESHOLD = 0.15  # 路径点到达阈值
TARGET_REACH_THRESHOLD = 0.2  # 目标点到达阈值

class SACContinuousNavigator:
    """SAC 连续导航系统主控制器"""
    
    def __init__(self, model_path: str, gui: bool = True, record: bool = False, use_llm: bool = True):
        """
        初始化 SAC 连续导航系统
        
        参数:
            model_path: 训练好的 SAC 模型路径
            gui: 是否显示GUI界面
            record: 是否录制视频
            use_llm: 是否使用 LLM 进行避障规划
        """
        self.model_path = model_path
        self.gui = gui
        self.record = record
        self.use_llm = use_llm
        
        # 系统状态
        self.running = False
        self.is_running = False
        self.paused = False
        self.exit_requested = False
        
        # 目标队列 - 实现连续导航 a->b->c
        self.target_queue: List = []
        self.current_target: Optional[List] = None
        self.target_reached = False  # 避免重复检测同一目标的到达
        self.home_position: List[float] = list(DEFAULT_INIT_POS)
        
        # 轨迹记录
        self.trajectory: List = []
        self.target_history: List = []
        self.llm_trajectory: Optional[np.ndarray] = None
        self.llm_trajectory_index = 0
        
        # 避障状态
        self.avoiding: bool = False
        self.avoidance_waypoint: Optional[np.ndarray] = None
        self.pre_avoid_target: Optional[List[float]] = None
        
        # 障碍物配置（动态生成）
        self.obstacles: List = []
        
        # 障碍物可视化
        self.obstacle_ids: List = []
        self.waypoint_marker_id: Optional[int] = None
        self.target_marker_id: Optional[int] = None
        
        # 统计信息
        self.stats: Dict = {
            'start_time': None,
            'targets_reached': 0,
            'commands_processed': 0,
            'total_distance': 0.0,
            'steps': 0,
            'avoidance_count': 0
        }
        
        # 核心组件（延迟初始化）
        self.env: Optional[ExtendedHoverAviary] = None
        self.model: Optional[SAC] = None
        self.keyboard_controller: Optional[KeyboardController] = None
        self.status_displayer: Optional[StatusDisplayer] = None
        
        # 网络服务器
        self.network_server: Optional[Any] = None
        self.network_thread: Optional[Any] = None
        self.network_enabled = True
    
    def initialize(self):
        """初始化所有系统组件"""
        print(f"[SAC导航器] 正在初始化系统组件...")
        
        # 1. 生成随机柱子位置
        self._generate_obstacles()
        
        # 2. 加载训练模型
        self._load_model()
        
        # 3. 创建扩展环境
        self._create_environment()
        
        # 4. 初始化控制器
        self._initialize_controllers()
        
        # 5. 添加障碍物和标记
        self._setup_obstacles()
        
        print(f"[SAC导航器] 系统初始化完成")
    
    def _generate_obstacles(self):
        """生成随机位置的柱子"""
        print(f"[障碍物] 正在生成随机柱子...")
        
        self.obstacles = generate_random_pillars(
            count=PILLAR_CONFIG['count'],
            radius=PILLAR_CONFIG['radius'],
            height=PILLAR_CONFIG['height'],
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            min_distance=0.8,
            origin_safe_radius=0.5
        )
        
        print(f"[障碍物] ✅ 成功生成 {len(self.obstacles)} 根柱子:")
        for i, pillar in enumerate(self.obstacles):
            print(f"  柱子 {i+1}: 位置=({pillar['pos'][0]:.2f}, {pillar['pos'][1]:.2f}), "
                  f"半径={pillar['radius']:.2f}m, 高度={pillar['height']:.2f}m")
        
    def _load_model(self):
        """加载训练好的 SAC 模型"""
        try:
            print(f"[模型加载] 正在加载 SAC 模型: {self.model_path}")
            self.model = SAC.load(self.model_path)
            print(f"[模型加载] ✅ SAC 模型加载成功")
        except Exception as e:
            print(f"[模型加载] ❌ 模型加载失败: {e}")
            raise
    
    def _create_environment(self):
        """创建扩展空间的测试环境"""
        try:
            print(f"[环境创建] 正在创建扩展环境...")
            
            # 使用默认起始位置和目标位置
            init_pos = np.array([DEFAULT_INIT_POS])
            init_rpy = np.array([[0, 0, 0]])
            
            # DEBUG: 打印配置值
            print(f"[DEBUG] DEFAULT_INIT_POS = {DEFAULT_INIT_POS}")
            print(f"[DEBUG] DEFAULT_TARGET_POS = {DEFAULT_TARGET_POS}")
            print(f"[DEBUG] init_pos = {init_pos}")
            
            self.env = ExtendedHoverAviary(
                initial_xyzs=init_pos,
                initial_rpys=init_rpy,
                gui=self.gui,
                record=self.record,
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                target_pos=DEFAULT_TARGET_POS,
                obstacles=True
            )
            
            self.current_target = DEFAULT_TARGET_POS.copy()
            print(f"[环境创建] ✅ 环境创建成功")
            print(f"[环境创建] 初始目标: {self.current_target}")
            
        except Exception as e:
            print(f"[环境创建] ❌ 环境创建失败: {e}")
            raise
    
    def _initialize_controllers(self):
        """初始化控制器组件"""
        try:
            # 初始化键盘控制器
            self.keyboard_controller = KeyboardController()
            
            # 初始化状态显示器
            self.status_displayer = StatusDisplayer(
                update_frequency=1.0/DISPLAY_CONFIG['update_frequency']
            )
            
            # 启动网络服务器
            if self.network_enabled:
                self.network_server = NetworkCommandServer(self)
                self.network_thread = threading.Thread(
                    target=self.network_server.start,
                    daemon=True
                )
                self.network_thread.start()
                
            print(f"[控制器] ✅ 控制器初始化成功")
            
        except Exception as e:
            print(f"[控制器] ❌ 控制器初始化失败: {e}")
            raise
    
    def _setup_obstacles(self):
        """在仿真环境中添加障碍物和标记"""
        if not self.gui or self.env is None:
            return
            
        try:
            client_id = self.env.CLIENT
            
            # 清空旧的障碍物 ID 列表（reset 后旧 ID 已失效）
            self.obstacle_ids.clear()
            
            # 添加柱子（使用生成的随机位置）
            for i, pillar in enumerate(self.obstacles):
                obs_id = self._add_obstacle(
                    client_id, 
                    pillar['pos'], 
                    pillar['radius'], 
                    pillar['height'],
                    color=PILLAR_CONFIG['color']
                )
                self.obstacle_ids.append(obs_id)
                print(f"  ✓ 柱子 {i+1} 已添加到仿真环境")
            
            # 添加目标标记
            self.target_marker_id = self._add_target_marker(client_id, self.current_target)
            
            print(f"[障碍物] ✅ 已在仿真中添加 {len(self.obstacle_ids)} 根柱子")
            
        except Exception as e:
            print(f"[障碍物] ⚠️ 添加障碍物失败: {e}")
    
    def _add_obstacle(self, client_id, pos, radius, height, color=[0.6, 0.6, 0.6, 1.0]):
        """在 PyBullet 仿真中添加圆柱体障碍物"""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=client_id
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=client_id
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[pos[0], pos[1], pos[2]],
            physicsClientId=client_id
        )
        return obstacle_id
    
    def _add_target_marker(self, client_id, pos, size=0.1, color=[0.2, 0.8, 0.2, 0.8]):
        """添加目标标记"""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color,
            physicsClientId=client_id
        )
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[pos[0], pos[1], pos[2]],
            physicsClientId=client_id
        )
        return marker_id
    
    def _add_waypoint_marker(self, client_id, pos, size=0.08, color=[0.2, 0.2, 0.8, 0.8]):
        """添加路径点标记"""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color,
            physicsClientId=client_id
        )
        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[pos[0], pos[1], pos[2]],
            physicsClientId=client_id
        )
        return marker_id
    
    def check_collision_risk(self, current_pos: np.ndarray) -> tuple:
        """
        检测是否有碰撞风险
        
        返回:
            (is_collision, obstacle_info) - 是否碰撞, 障碍物信息
        """
        for obs in self.obstacles:
            obs_pos = np.array(obs['pos'])
            distance_xy = np.linalg.norm(current_pos[:2] - obs_pos[:2])
            
            # 检查 XY 平面距离和 Z 轴高度
            if distance_xy < (obs['radius'] + COLLISION_THRESHOLD):
                # 只有在无人机高度低于柱子顶部时才需要避障
                # 柱子底部 z=0, 顶部 z=height, 中心 z=height/2
                pillar_top = obs['height']  # 柱子顶部高度
                pillar_bottom = 0.0         # 柱子底部高度
                
                # 如果无人机在柱子高度范围内，才需要避障
                if pillar_bottom - 0.1 < current_pos[2] < pillar_top + 0.1:
                    return True, obs
        
        return False, None
    
    def plan_avoidance_waypoint(self, current_pos: np.ndarray, target_pos: np.ndarray, obstacle_info: dict) -> np.ndarray:
        """
        规划避障路径点
        
        参数:
            current_pos: 当前位置
            target_pos: 目标位置
            obstacle_info: 障碍物信息
            
        返回:
            waypoint: 避障路径点
        """
        if self.use_llm and AVOIDANCE_AVAILABLE:
            try:
                # 使用 LLM 规划
                waypoint = plan_avoidance(
                    agent_pos=current_pos,
                    target_pos=target_pos,
                    obstacles=[obstacle_info],
                    bounds=EXTENDED_SPACE_CONFIG
                )
                if waypoint is not None:
                    print(f"[避障] 使用 LLM 规划路径点: {waypoint}")
                    return waypoint
            except Exception as e:
                print(f"[避障] LLM 规划失败: {e}, 使用几何规划")
        
        # 使用简单几何规划
        obs_pos = np.array(obstacle_info['pos'])
        obs_radius = obstacle_info['radius']
        
        # 计算垂直于障碍物-智能体连线的方向
        to_agent = current_pos - obs_pos
        to_agent_norm = to_agent / (np.linalg.norm(to_agent) + 1e-6)
        
        # 生成两个候选避障点（左右）
        perpendicular = np.array([-to_agent_norm[1], to_agent_norm[0], 0])
        
        safe_distance = obs_radius + COLLISION_THRESHOLD + 0.3
        candidate1 = obs_pos + perpendicular * safe_distance
        candidate2 = obs_pos - perpendicular * safe_distance
        
        # 选择距离目标更近的候选点
        dist1 = np.linalg.norm(candidate1 - target_pos)
        dist2 = np.linalg.norm(candidate2 - target_pos)
        
        waypoint = candidate1 if dist1 < dist2 else candidate2
        waypoint[2] = current_pos[2]  # 保持当前高度
        
        print(f"[避障] 使用几何规划路径点: {waypoint}")
        return waypoint
    
    def update_target_marker(self, new_target: List[float]):
        """更新目标标记位置"""
        if not self.gui or self.env is None:
            return
            
        try:
            client_id = self.env.CLIENT
            
            # 移除旧标记
            if self.target_marker_id is not None:
                p.removeBody(self.target_marker_id, physicsClientId=client_id)
            
            # 添加新标记
            self.target_marker_id = self._add_target_marker(client_id, new_target)
            
        except Exception as e:
            print(f"[标记] 更新目标标记失败: {e}")
    
    def start_navigation(self):
        """启动连续导航主循环"""
        if self.is_running:
            print("[导航] ⚠️ 系统已在运行中")
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        print("\n" + "="*60)
        print("[导航] 🚁 SAC 连续导航系统已启动")
        print("="*60)
        
        # 重置环境
        obs, info = self.env.reset(seed=42, options={})
        
        # 重新添加障碍物（reset 会清除所有物体）
        self._setup_obstacles()
        
        # 主循环
        try:
            while not self.exit_requested:
                start_time = time.time()
                
                # 1. 处理键盘输入
                self._handle_keyboard_input()
                
                # 2. 如果暂停，跳过主逻辑
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 3. 获取当前状态
                current_pos = self.env.pos[0]  # [x, y, z]
                self.trajectory.append(current_pos.copy())
                
                # 4. 检查是否有新目标
                if not self.target_queue and self.current_target is None:
                    # 等待新目标
                    time.sleep(0.1)
                    continue
                
                # 5. 如果当前无目标，从队列取下一个
                if self.current_target is None and self.target_queue:
                    self.current_target = self.target_queue.pop(0)
                    self.target_reached = False
                    self.update_target_marker(self.current_target)
                    print(f"[导航] 📍 新目标: {self.current_target}")
                
                # 6. 避障逻辑
                if not self.avoiding:
                    # 检查碰撞风险
                    is_collision, obstacle_info = self.check_collision_risk(current_pos)
                    
                    if is_collision:
                        # 进入避障模式
                        self.avoiding = True
                        self.pre_avoid_target = self.current_target.copy()
                        self.stats['avoidance_count'] += 1
                        
                        # 规划避障路径点
                        self.avoidance_waypoint = self.plan_avoidance_waypoint(
                            current_pos,
                            np.array(self.current_target),
                            obstacle_info
                        )
                        
                        # 添加路径点标记
                        if self.gui:
                            if self.waypoint_marker_id is not None:
                                p.removeBody(self.waypoint_marker_id, physicsClientId=self.env.CLIENT)
                            self.waypoint_marker_id = self._add_waypoint_marker(
                                self.env.CLIENT, 
                                self.avoidance_waypoint
                            )
                        
                        print(f"[避障] ⚠️ 检测到障碍物，进入避障模式")
                
                # 7. 确定当前导航目标
                if self.avoiding and self.avoidance_waypoint is not None:
                    nav_target = self.avoidance_waypoint
                    reach_threshold = WAYPOINT_REACH_THRESHOLD
                else:
                    nav_target = np.array(self.current_target)
                    reach_threshold = TARGET_REACH_THRESHOLD
                
                # 8. 更新环境目标（用于观测计算）
                self.env.TARGET_POS = nav_target.copy()
                
                # 9. 使用 SAC 模型推理动作
                action, _states = self.model.predict(obs, deterministic=True)
                
                # 10. 执行动作
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 10.5 检查是否被截断（失控保护）
                if truncated:
                    print(f"\n[安全] ⚠️ 检测到失控状态，正在重置...")
                    obs, info = self.env.reset(seed=42, options={})
                    self._setup_obstacles()  # 重新添加障碍物
                    print(f"[安全] ✅ 已重置，继续导航到目标: {self.current_target}")
                    continue  # 跳过本次循环的剩余部分
                
                # 11. 检查是否到达当前导航目标
                distance_to_target = np.linalg.norm(current_pos - nav_target)
                
                # DEBUG: 每100步打印一次详细信息
                if self.stats['steps'] % 100 == 0:
                    # 计算实际 RPM（使用训练时的系数 0.05）
                    rpm_values = self.env.HOVER_RPM * (1 + 0.05 * action[0])
                    print(f"\n[DEBUG] Step {self.stats['steps']}:")
                    print(f"  位置: {current_pos}  目标: {nav_target}  距离: {distance_to_target:.3f}m")
                    print(f"  动作值: {action[0]}  范围: [{action.min():.3f}, {action.max():.3f}]")
                    print(f"  RPM: [{rpm_values.min():.0f}, {rpm_values.max():.0f}]  (HOVER={self.env.HOVER_RPM:.0f})")
                
                if distance_to_target < reach_threshold:
                    if self.avoiding:
                        # 到达避障路径点，退出避障模式
                        self.avoiding = False
                        print(f"[避障] ✅ 到达避障路径点，恢复目标导航")
                        
                        # 移除路径点标记
                        if self.waypoint_marker_id is not None and self.gui:
                            p.removeBody(self.waypoint_marker_id, physicsClientId=self.env.CLIENT)
                            self.waypoint_marker_id = None
                        
                        self.avoidance_waypoint = None
                    else:
                        # 到达最终目标
                        if not self.target_reached:
                            self.target_reached = True
                            self.stats['targets_reached'] += 1
                            self.target_history.append(self.current_target.copy())
                            print(f"[导航] ✅ 到达目标 #{self.stats['targets_reached']}: {self.current_target}")
                            
                            # 准备下一个目标
                            self.current_target = None
                
                # 12. 更新统计
                self.stats['steps'] += 1
                
                # 13. 更新显示
                if self.status_displayer is not None:
                    status_info = {
                        'position': current_pos,
                        'target': self.current_target if self.current_target is not None else [0, 0, 0],
                        'distance': distance_to_target,
                        'targets_reached': self.stats['targets_reached'],
                        'steps': self.stats['steps'],
                        'avoidance_count': self.stats['avoidance_count'],
                        'mode': 'AVOIDING' if self.avoiding else 'NAVIGATING',
                        'paused': self.paused
                    }
                    
                    # 分离无人机状态和控制器状态
                    drone_state = {
                        'position': current_pos,
                        'velocity': obs[10:13] if len(obs) > 13 else [0, 0, 0],
                        'orientation': obs[7:10] if len(obs) > 10 else [0, 0, 0],
                        'target_position': self.current_target if self.current_target is not None else [0, 0, 0],
                        'distance_to_target': distance_to_target
                    }
                    controller_status = {
                        'target': self.current_target if self.current_target is not None else [0, 0, 0],
                        'distance': distance_to_target,
                        'mode': 'AVOIDING' if self.avoiding else 'NAVIGATING',
                        'is_paused': self.paused,
                        'targets_reached': self.stats['targets_reached']
                    }
                    self.status_displayer.update_display(drone_state, controller_status)
                
                # 14. 同步
                if self.gui:
                    self.env.render()
                
                # 15. 控制帧率
                elapsed = time.time() - start_time
                if elapsed < self.env.CTRL_TIMESTEP:
                    time.sleep(self.env.CTRL_TIMESTEP - elapsed)
        
        except KeyboardInterrupt:
            print("\n[导航] 接收到中断信号，正在关闭...")
        
        finally:
            self._shutdown()
    
    def _handle_keyboard_input(self):
        """处理键盘输入"""
        if self.keyboard_controller is None:
            return
        
        cmd = self.keyboard_controller.get_command()
        
        if cmd is not None:
            if cmd['type'] == 'pause':
                self.paused = not self.paused
                print(f"[控制] {'⏸ 已暂停' if self.paused else '▶ 已继续'}")
            
            elif cmd['type'] == 'quit':
                self.exit_requested = True
                print("[控制] 🛑 请求退出")
            
            elif cmd['type'] == 'home':
                self.add_target(self.home_position)
                print(f"[控制] 🏠 返回起点: {self.home_position}")
            
            elif cmd['type'] == 'target':
                target = cmd['target']
                self.add_target(target)
                print(f"[控制] 📍 新目标: {target}")
    
    def add_target(self, target: List[float]):
        """添加新目标到队列"""
        self.target_queue.append(target)
        self.stats['commands_processed'] += 1
        print(f"[队列] 添加目标: {target}, 队列长度: {len(self.target_queue)}")
    
    def _shutdown(self):
        """关闭系统，清理资源"""
        print("\n[关闭] 正在清理资源...")
        
        # 打印统计信息
        if self.stats['start_time'] is not None:
            elapsed = time.time() - self.stats['start_time']
            print("\n" + "="*60)
            print("📊 运行统计")
            print("="*60)
            print(f"运行时长: {elapsed:.2f} 秒")
            print(f"到达目标数: {self.stats['targets_reached']}")
            print(f"总步数: {self.stats['steps']}")
            print(f"避障次数: {self.stats['avoidance_count']}")
            print(f"处理命令数: {self.stats['commands_processed']}")
            print("="*60)
        
        # 可视化轨迹
        if VISUALIZATION_AVAILABLE and len(self.trajectory) > 0:
            self._visualize_trajectory()
        
        # 关闭环境
        if self.env is not None:
            self.env.close()
        
        self.is_running = False
        print("[关闭] ✅ 系统已关闭")
    
    def _visualize_trajectory(self):
        """可视化飞行轨迹"""
        try:
            trajectory_array = np.array(self.trajectory)
            
            fig = plt.figure(figsize=(15, 5))
            
            # 3D 轨迹图
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                     'b-', linewidth=2, label='飞行轨迹')
            
            # 绘制目标点
            if self.target_history:
                targets = np.array(self.target_history)
                ax1.scatter(targets[:, 0], targets[:, 1], targets[:, 2], 
                           c='g', marker='*', s=200, label='到达目标')
            
            # 绘制柱子（障碍物）
            for obs in self.obstacles:
                pos = obs['pos']
                ax1.scatter(pos[0], pos[1], pos[2], c='gray', marker='s', s=150, alpha=0.6, label='柱子' if obs == self.obstacles[0] else '')
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D 飞行轨迹')
            ax1.legend()
            ax1.grid(True)
            
            # XY 平面投影
            ax2 = fig.add_subplot(132)
            ax2.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', linewidth=2)
            if self.target_history:
                targets = np.array(self.target_history)
                ax2.scatter(targets[:, 0], targets[:, 1], c='g', marker='*', s=200)
            # 绘制柱子投影（圆圈）
            for obs in self.obstacles:
                from matplotlib.patches import Circle
                circle = Circle((obs['pos'][0], obs['pos'][1]), obs['radius'], 
                                   color='gray', alpha=0.4, linewidth=2, edgecolor='black')
                ax2.add_patch(circle)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('XY 平面投影')
            ax2.grid(True)
            ax2.axis('equal')
            
            # 高度随时间变化
            ax3 = fig.add_subplot(133)
            time_steps = np.arange(len(trajectory_array)) * 0.02  # 假设 50Hz
            ax3.plot(time_steps, trajectory_array[:, 2], 'b-', linewidth=2)
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('高度 Z (m)')
            ax3.set_title('高度变化')
            ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig('sac_continuous_navigation_trajectory.png', dpi=150)
            print(f"[可视化] ✅ 轨迹图已保存: sac_continuous_navigation_trajectory.png")
            
        except Exception as e:
            print(f"[可视化] ⚠️ 轨迹可视化失败: {e}")


class NetworkCommandServer:
    """网络命令服务器（与 PPO 版本兼容）"""
    
    def __init__(self, navigator: SACContinuousNavigator):
        self.navigator = navigator
        self.host = NETWORK_CONFIG['host']
        self.port = NETWORK_CONFIG['port']
        self.socket = None
        
    def start(self):
        """启动服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            print(f"[网络] 📡 命令服务器已启动: {self.host}:{self.port}")
            
            while not self.navigator.exit_requested:
                try:
                    self.socket.settimeout(1.0)
                    client, address = self.socket.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(client, address),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"[网络] ❌ 服务器错误: {e}")
        finally:
            if self.socket:
                self.socket.close()
    
    def _handle_client(self, client_socket, address):
        """处理客户端连接"""
        try:
            data = client_socket.recv(1024).decode('utf-8')
            command = json.loads(data)
            
            if command['type'] == 'target':
                target = command['target']
                self.navigator.add_target(target)
                response = {'status': 'success', 'message': f'目标已添加: {target}'}
            
            elif command['type'] == 'home':
                self.navigator.add_target(self.navigator.home_position)
                response = {'status': 'success', 'message': '返回起点'}
            
            elif command['type'] == 'stop':
                # 停止飞行：清空目标队列并设置当前位置为目标
                self.navigator.target_queue.clear()
                current_pos = self.navigator.env.pos[0]
                self.navigator.current_target = current_pos.copy()
                self.navigator.avoidance_waypoint = None  # 清除避障路径点
                response = {'status': 'success', 'message': f'已停止飞行，悬停在当前位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]'}
                print(f"[导航] 🛑 停止飞行命令已执行")
            
            else:
                response = {'status': 'error', 'message': '未知命令'}
            
            client_socket.send(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"[网络] 处理客户端请求失败: {e}")
        finally:
            client_socket.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC 连续导航系统')
    parser.add_argument('--model_path', type=str, required=True,
                       help='SAC 模型路径')
    parser.add_argument('--gui', type=str, default='true',
                       help='是否显示 GUI')
    parser.add_argument('--record', type=str, default='false',
                       help='是否录制视频')
    parser.add_argument('--use_llm', type=str, default='true',
                       help='是否使用 LLM 避障')
    
    args = parser.parse_args()
    
    # 转换参数
    from gym_pybullet_drones.utils.utils import str2bool
    gui = str2bool(args.gui)
    record = str2bool(args.record)
    use_llm = str2bool(args.use_llm)
    
    # 创建导航器
    navigator = SACContinuousNavigator(
        model_path=args.model_path,
        gui=gui,
        record=record,
        use_llm=use_llm
    )
    
    # 初始化
    navigator.initialize()
    
    # 启动导航
    navigator.start_navigation()


if __name__ == '__main__':
    main()
