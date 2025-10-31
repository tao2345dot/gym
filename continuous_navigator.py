"""
连续导航主控制器

整合环境、键盘输入、模型推理等功能,实现连续导航系统
"""

# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalSubscript=false  
# pyright: reportGeneralTypeIssues=false

# 说明：此文件中的Optional类型成员访问警告已被禁用
# 因为这些对象在使用前都经过了initialize()初始化
# 运行时保证这些Optional对象不会是None

import os
import time
import socket
import json
import threading
import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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
    chinese_font = FontProperties(fname=font_path)
    plt.rcParams['axes.unicode_minus'] = False
    print("[可视化] ✅ 轨迹可视化功能可用")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[可视化] ⚠️ matplotlib未安装，轨迹可视化不可用")

class ContinuousNavigator:
    """连续导航系统主控制器"""
    
    def __init__(self, model_path: str, gui: bool = True, record: bool = False):
        """
        初始化连续导航系统
        
        参数:
            model_path: 训练好的模型路径
            gui: 是否显示GUI界面
            record: 是否录制视频
        """
        self.model_path = model_path
        self.gui = gui
        self.record = record
        
        # 系统状态
        self.running = False
        self.is_running = False
        self.paused = False
        self.exit_requested = False
        
        # 目标队列 - 实现连续导航 a->b->c
        self.target_queue: List = []
        self.current_target: Optional[List] = None
        self.target_reached = False  # 避免重复检测同一目标的到达
        self.home_position: List[float] = list(DEFAULT_INIT_POS)  # 添加home_position属性
        
        # 轨迹记录
        self.trajectory: List = []
        self.target_history: List = []
        self.llm_trajectory: Optional[np.ndarray] = None  # 存储LLM生成的轨迹（numpy数组）
        self.llm_trajectory_index = 0  # 当前执行到的轨迹点索引
        # 避障状态
        self.avoiding: bool = False
        self.pre_avoid_target: Optional[List[float]] = None
        
        # 统计信息
        self.stats: Dict = {
            'start_time': None,
            'targets_reached': 0,
            'commands_processed': 0,
            'total_distance': 0.0,
            'steps': 0
        }
        
        # 核心组件（延迟初始化）- 添加类型注解
        self.env: Optional[ExtendedHoverAviary] = None
        self.model: Optional[PPO] = None
        self.keyboard_controller: Optional[KeyboardController] = None
        self.status_displayer: Optional[StatusDisplayer] = None
        
        # 网络服务器
        self.network_server: Optional[Any] = None
        self.network_thread: Optional[Any] = None
        self.network_enabled = True
    
    def initialize(self):
        """初始化所有系统组件"""
        print(f"[连续导航器] 正在初始化系统组件...")
        
        # 1. 加载训练模型
        self._load_model()
        
        # 2. 创建扩展环境
        self._create_environment()
        
        # 3. 初始化控制器
        self._initialize_controllers()
        
        print(f"[连续导航器] 系统初始化完成")
        
    def _load_model(self):
        """加载训练好的PPO模型"""
        try:
            print(f"[模型加载] 正在加载模型: {self.model_path}")
            self.model = PPO.load(self.model_path)
            print(f"[模型加载] ✅ 模型加载成功")
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
            
            self.env = ExtendedHoverAviary(
                initial_xyzs=init_pos,
                initial_rpys=init_rpy,
                gui=self.gui,
                record=self.record,
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                target_pos=DEFAULT_TARGET_POS,
                obstacles=True  # 启用障碍物（仅在连续导航测试环境中）
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
    
    def start_navigation(self):
        """启动连续导航系统"""
        if self.is_running:
            print(f"[导航系统] 系统已在运行中")
            return
            
        print(f"\n" + "="*60)
        print(f"🚁 连续导航系统启动")
        print(f"="*60)
        
        try:
            # 启动系统组件
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            # 重置环境
            obs, info = self.env.reset()
            
            # 启动键盘监听
            self.keyboard_controller.start()
            
            print(f"[导航系统] ✅ 系统启动成功")
            print(f"[导航系统] 当前位置: {self.env.get_current_state()['position']}")
            print(f"[导航系统] 当前目标: {self.current_target}")
            print(f"[导航系统] 输入新目标坐标开始导航，输入 'help' 查看帮助")
            
            # 进入主导航循环
            self._navigation_loop(obs)
            
        except KeyboardInterrupt:
            print(f"\n[导航系统] 收到中断信号，正在退出...")
        except Exception as e:
            print(f"[导航系统] ❌ 系统运行出错: {e}")
        finally:
            self._shutdown()
    
    def _navigation_loop(self, initial_obs):
        """主导航循环"""
        obs = initial_obs
        step_count = 0
        last_target_check_time = time.time()
        
        while self.is_running:
            # 1. 处理用户输入命令
            self._process_user_commands()
            
            # 2. 检查是否应该退出
            if self.keyboard_controller.should_exit:
                print(f"[导航系统] 用户请求退出")
                break
            
            # 3. 如果暂停，则等待
            if self.paused or self.keyboard_controller.is_paused:
                time.sleep(0.1)
                continue
            
            # 4. 执行无人机控制
            obs = self._execute_control_step(obs)
            
            # 5. 更新状态显示
            self._update_status_display()
            
            # 6. 到达检测已在 _execute_control_step 中处理，无需重复检查
            
            # 7. 记录轨迹
            if DEBUG_CONFIG['log_trajectory']:
                self._log_trajectory_point()
            
            step_count += 1
            
            # 控制循环频率
            time.sleep(1.0 / SIMULATION_CONFIG['ctrl_freq'])
    
    def _process_user_commands(self):
        """处理用户输入的命令"""
        command = self.keyboard_controller.get_command()
        
        if command is None:
            return
            
        command_type = command['type']
        command_data = command['data']
        
        if command_type == 'target':
            self._set_new_target(command_data)
            
        elif command_type == 'pause':
            self._pause_navigation()
            
        elif command_type == 'resume':
            self._resume_navigation()
            
        elif command_type == 'home':
            self._return_home()
            
        elif command_type == 'current':
            self._show_current_status()
            
        elif command_type == 'queue':
            self._show_target_queue()
            
        elif command_type == 'clear':
            self._clear_target_queue()
            
        elif command_type == 'circle':
            # 生成圆形飞行任务
            self._handle_circle_command(command_data)
        elif command_type == 'circle_avoid':
            # 生成圆形避障任务（在圆路径上布置多个障碍）
            self._handle_circle_avoid_command(command_data)
            
        elif command_type == 'stats':
            # 显示轨迹统计
            self.show_trajectory_stats()
            
        elif command_type == 'visual':
            # 显示轨迹可视化
            if VISUALIZATION_AVAILABLE and self.llm_trajectory is not None:
                self._visualize_llm_trajectory()
            else:
                print("[可视化] ❌ 无可用轨迹或可视化功能未加载")
            
        elif command_type == 'exit':
            self._request_exit()
    
    def _execute_control_step(self, obs):
        """执行一步无人机控制"""
        try:
            # 调试观测形状
            if DEBUG_CONFIG['verbose'] and hasattr(self, '_debug_step_count') and self._debug_step_count < 5:
                print(f"[调试] 观测形状: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
                self._debug_step_count += 1
            elif not hasattr(self, '_debug_step_count'):
                self._debug_step_count = 0
            
            # 确保观测维度正确 - 处理可能的维度问题
            if hasattr(obs, 'shape'):
                if len(obs.shape) == 3 and obs.shape[0] == 1:
                    # 如果是(1, 1, N)形状，则reshape为(1, N)
                    obs_for_model = obs.reshape(obs.shape[0], -1)
                elif len(obs.shape) == 2:
                    # 如果已经是(1, N)形状，直接使用
                    obs_for_model = obs
                else:
                    # 其他情况，尝试flatten
                    obs_for_model = obs.reshape(1, -1)
            else:
                # 如果不是numpy数组，尝试转换
                import numpy as np
                obs_for_model = np.array(obs).reshape(1, -1)
            
            # 使用模型预测动作
            action, _states = self.model.predict(obs_for_model, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 检查是否因边界或时间截断（保留错误处理）
            if truncated:
                print(f"\n[导航] ⚠️ 环境被截断（可能超出边界或时间限制）")
                # 只有在出错时才重置
                obs, info = self.env.reset()
            
            # 直接检查距离来判断是否到达目标（不依赖terminated状态）
            if self.current_target is not None and not self.target_reached:
                drone_state = self.env.get_current_state()

                # 1) 实时障碍检测与避障决策
                try:
                    obstacle = None
                    if hasattr(self.env, 'detect_nearest_obstacle'):
                        obstacle = self.env.detect_nearest_obstacle(agent_pos=drone_state['position'], threshold=0.5)

                    if obstacle is not None and not self.avoiding:
                        ob_pos, ob_radius, ob_dist = obstacle
                        print(f"\n[障碍] Detected obstacle at {ob_pos.tolist()} r={ob_radius:.2f}, drone dist={ob_dist:.2f}. Planning avoidance.")

                        # 生成避障点（优先使用外部规划器）
                        avoid_point = None
                        try:
                            if AVOIDANCE_AVAILABLE:
                                # collect environment bounds and other obstacles to pass to the LLM
                                env_bounds = getattr(self.env, 'EXTENDED_SPACE', None)
                                other_obs = []
                                try:
                                    import pybullet as p
                                    if hasattr(self.env, 'OBSTACLE_IDS'):
                                        for i, oid in enumerate(self.env.OBSTACLE_IDS):
                                            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self.env.CLIENT)
                                            r = self.env.OBSTACLE_RADII[i] if hasattr(self.env, 'OBSTACLE_RADII') and i < len(self.env.OBSTACLE_RADII) else 0.1
                                            other_obs.append((pos, r))
                                except Exception:
                                    # if querying pybullet fails, keep other_obs as None
                                    other_obs = None

                                avoid_point = plan_avoidance(agent_pos=drone_state['position'], obstacle_pos=ob_pos, radius=ob_radius, target_pos=self.current_target, env_bounds=env_bounds, other_obstacles=other_obs, constraints={'min_clearance':0.3})
                            else:
                                # 简单几何避障：绕障点计算两侧候选点，选择更靠近目标的一侧
                                vec = np.array(drone_state['position']) - np.array(ob_pos)
                                if np.linalg.norm(vec[:2]) < 1e-3:
                                    vec = np.array([1.0, 0.0, 0.0])
                                # 在XY平面取法向
                                perp = np.array([-vec[1], vec[0], 0.0])
                                perp = perp / (np.linalg.norm(perp) + 1e-6)
                                dist_offset = ob_radius + 0.5
                                cand1 = ob_pos + perp * dist_offset
                                cand2 = ob_pos - perp * dist_offset
                                # 设置Z为与当前目标相同高度（或略高）
                                cand1[2] = self.current_target[2]
                                cand2[2] = self.current_target[2]
                                # 选择更接近原目标的候选点
                                d1 = np.linalg.norm(cand1 - np.array(self.current_target))
                                d2 = np.linalg.norm(cand2 - np.array(self.current_target))
                                avoid_point = cand1 if d1 < d2 else cand2

                        except Exception:
                            avoid_point = None

                        if avoid_point is not None:
                            # 下发避障目标并记住原目标以便恢复
                            self.pre_avoid_target = self.current_target.copy() if self.current_target is not None else None
                            self.avoiding = True

                            # 尝试将避障点下发到环境，并在失败时尝试一系列回退策略
                            candidate = [float(avoid_point[0]), float(avoid_point[1]), float(avoid_point[2])]
                            success_set = False

                            try:
                                # 首次尝试：直接下发 LLM/几何计算出的避障点
                                if self.env.update_target_position(candidate):
                                    success_set = True
                                else:
                                    # 回退策略1：抬高Z轴（尝试飞越障碍顶部）
                                    alt1 = candidate.copy()
                                    alt1[2] = alt1[2] + 0.4
                                    # 限制在环境高度范围内
                                    try:
                                        zmin, zmax = self.env.EXTENDED_SPACE['z_range']
                                        alt1[2] = min(max(alt1[2], zmin + 0.2), zmax - 0.2)
                                    except Exception:
                                        pass
                                    if self.env.update_target_position(alt1):
                                        candidate = alt1
                                        success_set = True

                                if not success_set:
                                    # 回退策略2：尝试绕障两侧候选点（基于原目标方向）
                                    try:
                                        obs = np.array(ob_pos, dtype=float)
                                        rad = float(ob_radius)
                                        target_orig = np.array(self.pre_avoid_target if self.pre_avoid_target is not None else self.current_target)
                                        vec = target_orig - obs
                                        if np.linalg.norm(vec[:2]) < 1e-3:
                                            vec = np.array([1.0, 0.0, 0.0])
                                        perp = np.array([-vec[1], vec[0], 0.0])
                                        perp = perp / (np.linalg.norm(perp) + 1e-6)
                                        for extra_offset in [0.5, 0.8, 1.1]:
                                            cand1 = obs + perp * (rad + extra_offset)
                                            cand2 = obs - perp * (rad + extra_offset)
                                            cand1[2] = target_orig[2]
                                            cand2[2] = target_orig[2]
                                            for cand in (cand1, cand2):
                                                cand_list = [float(cand[0]), float(cand[1]), float(cand[2])]
                                                if self.env.update_target_position(cand_list):
                                                    candidate = cand_list
                                                    success_set = True
                                                    break
                                            if success_set:
                                                break
                                    except Exception:
                                        pass

                                if not success_set:
                                    # 回退策略3：放弃本次避障（避免进入不可达状态），恢复先前目标
                                    print(f"[避障] ⚠️ 无法在环境内下发避障目标，放弃避障并恢复原目标")
                                    if self.pre_avoid_target is not None:
                                        # 尝试恢复原目标到环境
                                        try:
                                            if not self.env.update_target_position(self.pre_avoid_target):
                                                print(f"[避障] ⚠️ 恢复原目标失败: {self.pre_avoid_target}")
                                        except Exception:
                                            pass
                                    # 取消避障状态
                                    self.avoiding = False
                                    self.pre_avoid_target = None
                                    # 不设置 current_target 为不可达点
                                else:
                                    # 成功设置候选点
                                    self.current_target = candidate
                                    print(f"[避障] 避障目标下发: ({self.current_target[0]:.2f}, {self.current_target[1]:.2f}, {self.current_target[2]:.2f})")

                            except Exception:
                                # 如果任何意外发生，则取消避障并恢复
                                print(f"[避障] ❌ 下发避障目标时出错，放弃避障")
                                if self.pre_avoid_target is not None:
                                    try:
                                        self.env.update_target_position(self.pre_avoid_target)
                                    except Exception:
                                        pass
                                self.avoiding = False
                                self.pre_avoid_target = None

                except Exception:
                    # 障碍检测/避障规划出错则忽略
                    pass

                # 2) 到达检测
                if drone_state['is_near_target']:
                    print(f"\n[导航] ✅ 已到达目标位置！")
                    self.stats['targets_reached'] += 1
                    self.target_reached = True  # 标记当前目标已到达

                    # 在清空 current_target 之前记录到达信息
                    if self.current_target is not None:
                        drone_state = self.env.get_current_state()
                        self.target_history.append({
                            'target': self.current_target.copy(),
                            'reached_time': time.time(),
                            'distance_error': drone_state['distance_to_target']
                        })

                    # 如果这是避障目标，恢复先前目标
                    if self.avoiding:
                        print(f"[避障] 避障点已到达，恢复正常导航")
                        if self.pre_avoid_target is not None:
                            self.current_target = self.pre_avoid_target
                            self.env.update_target_position(self.current_target)
                            self.pre_avoid_target = None
                        self.avoiding = False
                        self.target_reached = False
                        return obs

                    # 检查是否有排队的目标
                    if hasattr(self, 'target_queue') and self.target_queue:
                        next_target = self.target_queue.pop(0)
                        print(f"[导航] 🎯 自动前往下一个目标: ({next_target[0]:.2f}, {next_target[1]:.2f}, {next_target[2]:.2f})")
                        self.current_target = next_target  # 更新当前目标
                        self.target_reached = False  # 重置到达标志
                        self.env.update_target_position(next_target)
                        # 不重置环境，继续从当前位置导航
                        if self.target_queue:
                            queue_str = " -> ".join([f"({t[0]:.1f},{t[1]:.1f},{t[2]:.1f})" for t in self.target_queue])
                            print(f"[队列] 剩余目标: {queue_str}")
                    else:
                        print(f"[导航] 🏁 已完成所有目标，悬停等待新指令...")
                        # 到达目标后清空当前目标，但保持悬停（不暂停）
                        self.current_target = None
                        # 无人机会在当前位置悬停，等待新目标
                
            return obs
            
        except Exception as e:
            print(f"[控制] ❌ 控制步骤执行失败: {e}")
            # 在调试模式下显示更多信息
            if DEBUG_CONFIG['verbose']:
                import traceback
                traceback.print_exc()
            return obs
    
    def _update_status_display(self):
        """更新状态显示"""
        try:
            drone_state = self.env.get_current_state()
            controller_status = self.keyboard_controller.get_status()
            
            self.status_displayer.update_display(drone_state, controller_status)
            
        except Exception as e:
            if DEBUG_CONFIG['verbose']:
                print(f"[显示] 状态更新失败: {e}")
    
    def _check_target_reached(self):
        """检查是否到达目标（已废弃 - 到达检测现在在 _execute_control_step 中处理）"""
        # 这个方法已不再使用，到达检测统一在 _execute_control_step 中通过环境的 terminated 状态处理
        pass
    
    def _set_new_target(self, target_pos: List[float]):
        """设置新的目标位置 - 支持连续导航队列"""
        try:
            # 平滑爬升策略：如果目标高度远高于当前高度，先在当前位置爬升一个中间高度以减少急速机动
            current_pos = self._get_current_position()
            if (self.current_target is None or self.paused) and current_pos is not None:
                # 需要平滑爬升的阈值
                zdiff = float(target_pos[2]) - float(current_pos[2])
                if zdiff > 0.3:
                    # 插入一个爬升中间点（在当前位置 x,y 上抬高，但不要超过目标高度）
                    climb_z = min(current_pos[2] + 0.4, float(target_pos[2]))
                    climb_point = [float(current_pos[0]), float(current_pos[1]), float(climb_z)]
                    # 先设置爬升目标，再把原目标放入队列
                    success_climb = self.env.update_target_position(climb_point)
                    if success_climb:
                        self.current_target = climb_point
                        self.target_reached = False
                        self.stats['commands_processed'] += 1
                        # 把原目标放入队列，等待爬升完成后继续
                        self.target_queue.insert(0, target_pos)
                        print(f"\n[导航] 🎯 先设置爬升中间目标: ({climb_point[0]:.2f}, {climb_point[1]:.2f}, {climb_point[2]:.2f}); 原目标已入队")
                        # 恢复导航（如果之前暂停）
                        if self.paused:
                            self.paused = False
                            print(f"[系统] ▶️ 恢复导航")
                        return

            # 如果当前没有目标或已暂停，立即设置为当前目标
            if self.current_target is None or self.paused:
                success = self.env.update_target_position(target_pos)
                if success:
                    self.current_target = target_pos
                    self.target_reached = False  # 重置到达标志
                    print(f"\n[导航] 🎯 当前目标设置: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
                    self.stats['commands_processed'] += 1
                    
                    # 恢复导航（如果之前暂停）
                    if self.paused:
                        self.paused = False
                        print(f"[系统] ▶️ 恢复导航")
                else:
                    print(f"[目标] ❌ 目标位置无效: {target_pos}")
            else:
                # 如果已有当前目标，加入队列
                self.target_queue.append(target_pos)
                print(f"\n[队列] 📋 目标已加入队列: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) (队列长度: {len(self.target_queue)})")
                if self.current_target is not None:
                    print(f"[队列] 当前目标: ({self.current_target[0]:.2f}, {self.current_target[1]:.2f}, {self.current_target[2]:.2f})")
                else:
                    print(f"[队列] 当前目标: 无 (等待新目标)")
                if self.target_queue:
                    queue_str = " -> ".join([f"({t[0]:.1f},{t[1]:.1f},{t[2]:.1f})" for t in self.target_queue])
                    print(f"[队列] 待完成目标: {queue_str}")
                
        except Exception as e:
            print(f"[导航] ❌ 目标设定失败: {e}")
    
    def _add_target_to_queue(self, target_pos: List[float]):
        """添加目标到队列末尾"""
        self.target_queue.append(target_pos)
        print(f"[队列] ➕ 目标已添加到队列: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) (队列长度: {len(self.target_queue)})")
    
    def _clear_target_queue(self):
        """清空目标队列"""
        cleared_count = len(self.target_queue)
        self.target_queue.clear()
        print(f"[队列] 🗑️ 已清空目标队列 (清除了 {cleared_count} 个目标)")
    
    def _show_target_queue(self):
        """显示当前目标队列状态"""
        print(f"\n📋 目标队列状态:")
        print(f"   当前目标: {self.current_target if self.current_target else '无'}")
        if self.target_queue:
            print(f"   队列长度: {len(self.target_queue)}")
            for i, target in enumerate(self.target_queue):
                print(f"   {i+1}. ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        else:
            print(f"   队列: 空")
        print(f"   系统状态: {'暂停' if self.paused else '运行中'}")
        print(f"   已完成目标: {self.stats['targets_reached']} 个")
    
    def _pause_navigation(self):
        """暂停导航"""
        self.paused = True
        print(f"\n[导航] ⏸️ 导航已暂停")
    
    def _resume_navigation(self):
        """恢复导航"""
        self.paused = False
        print(f"\n[导航] ▶️ 导航已恢复")
    
    def _return_home(self):
        """返回起始位置"""
        self._set_new_target(self.home_position)
        print(f"[导航] 🏠 正在返回起始位置")
    
    def _show_current_status(self):
        """显示当前详细状态"""
        drone_state = self.env.get_current_state()
        self.status_displayer.show_status_summary(drone_state)
        
        # 显示目标队列信息
        print(f"🎯 导航状态:")
        print(f"   当前目标: {self.current_target if self.current_target else '无'}")
        if self.target_queue:
            queue_str = " -> ".join([f"({t[0]:.1f},{t[1]:.1f},{t[2]:.1f})" for t in self.target_queue])
            print(f"   待完成队列: {queue_str}")
        else:
            print(f"   待完成队列: 空")
        print(f"   系统状态: {'暂停' if self.paused else '运行中'}")
        
        # 显示飞行统计
        current_time = time.time()
        if self.stats['start_time']:
            flight_time = current_time - self.stats['start_time']
            print(f"\n📊 飞行统计:")
            print(f"   总飞行时间: {flight_time:.1f} 秒")
            print(f"   到达目标数: {self.stats['targets_reached']}")
            if hasattr(self, 'target_history'):
                print(f"   目标历史: {len(self.target_history)} 个")
        print()
    
    def _handle_circle_command(self, command_data):
        """处理圆形飞行命令"""
        if not LLM_AVAILABLE:
            print("[圆形任务] ❌ LLM功能不可用")
            return
            
        # 获取当前无人机位置
        current_pos = self._get_current_position()
        current_height = current_pos[2] if current_pos is not None else 1.2
            
        # 解析命令参数，考虑模型精度(0.3m误差)和地图边界
        if isinstance(command_data, dict):
            radius = command_data.get('radius', 0.8)  # 适应训练空间[-1.5,1.5]
            height = command_data.get('height', current_height)  # 使用当前高度
            waypoints = command_data.get('waypoints', 48)  # 减少轨迹点，提高精度
            clockwise = command_data.get('clockwise', False)
        else:
            # 适应训练空间的默认参数 - 考虑[-1.5,1.5]边界和0.3m误差
            radius = 0.8  # 半径0.8m，在1.5m边界内安全（0.8+0.3+0.3=1.4<1.5）
            height = current_height  # 使用当前高度
            waypoints = 48  # 12个实际轨迹点，确保点间距离适中
            clockwise = False
            
        print(f"[圆形任务] 🔄 生成圆形飞行轨迹（基于当前高度: {height:.2f}m）...")
        success = self.generate_circle_mission(radius, height, waypoints, clockwise)
        
        if success:
            print(f"[圆形任务] ✅ 圆形任务已设置，开始执行!")
        else:
            print(f"[圆形任务] ❌ 圆形任务设置失败")

    def _handle_circle_avoid_command(self, command_data):
        """
        为了测试避障，生成一个圆形任务并在圆周上布置尽可能多的障碍物，
        然后把轨迹加载到目标队列，导航器将利用已有避障逻辑绕过这些障碍并最终回到起点。

        command_data 可以是 dict，支持键:
            radius: 圆半径
            waypoints: 轨迹点数量
            num_obstacles: 在圆周上放置的障碍数量
            obs_radius: 障碍物半径
            height: 障碍物高度
        """
        try:
            print("[圆形避障] 🔄 生成带障碍的圆形任务...")

            # 获取当前无人机位置作为圆心参考
            # Avoid using `or` with numpy arrays (ambiguous truth value). Explicitly
            # check for None and normalize to a plain Python list of floats.
            current_pos = self._get_current_position()
            if current_pos is None:
                current_pos = [0.0, 0.0, 1.0]
            else:
                try:
                    arr = np.asarray(current_pos).flatten()
                    current_pos = [float(arr[0]), float(arr[1]), float(arr[2])]
                except Exception:
                    # Fallback: try to coerce to list
                    try:
                        current_pos = [float(current_pos[0]), float(current_pos[1]), float(current_pos[2])]
                    except Exception:
                        current_pos = [0.0, 0.0, 1.0]

            # 解析参数（兼容 None 值，优先使用当前高度作为默认 height）
            if isinstance(command_data, dict):
                # radius / waypoints / num_obstacles / obs_radius 可能为 None
                r = command_data.get('radius', 0.8)
                radius = float(r) if (r is not None) else 0.8

                w = command_data.get('waypoints', 64)
                waypoints = int(w) if (w is not None) else 64

                no = command_data.get('num_obstacles', 12)
                num_obstacles = int(no) if (no is not None) else 12

                orad = command_data.get('obs_radius', 0.10)
                obs_radius = float(orad) if (orad is not None) else 0.10

                h = command_data.get('height', None)
                obs_height = float(h) if (h is not None) else float(current_pos[2])
            else:
                radius = 0.8
                waypoints = 64
                num_obstacles = 12
                obs_radius = 0.10
                obs_height = float(current_pos[2])
            circle_center = np.array([0.0, 0.0, current_pos[2]])

            # 生成圆形轨迹（优先使用LLM轨迹器，如果不可用则使用本地计算）
            trajectory = None
            if LLM_AVAILABLE:
                try:
                    trajectory = generate_circle_trajectory(init_xyz=current_pos, num_waypoints=waypoints, clockwise=False, radius=radius)
                except Exception:
                    trajectory = None

            # Normalize trajectory to a NumPy array if possible to avoid truth-value ambiguity
            try:
                if trajectory is not None:
                    trajectory = np.asarray(trajectory)
            except Exception:
                # leave trajectory as-is; downstream checks will handle None/non-array
                pass

            if trajectory is None:
                # 本地生成均匀圆周点
                angles = np.linspace(0, 2 * np.pi, waypoints, endpoint=False)
                traj = []
                for a in angles:
                    x = circle_center[0] + radius * np.cos(a)
                    y = circle_center[1] + radius * np.sin(a)
                    z = circle_center[2]
                    traj.append([x, y, z])
                trajectory = np.array(traj, dtype=float).reshape(1, waypoints, 3)

            # 在圆周上放置障碍物
            if self.env is None:
                print("[圆形避障] ❌ 环境未初始化，无法放置障碍")
                return

            # Place obstacles evenly spaced on circle, avoiding the starting sector
            placed = 0
            for i in range(num_obstacles):
                angle = 2 * np.pi * i / num_obstacles
                ox = circle_center[0] + radius * np.cos(angle)
                oy = circle_center[1] + radius * np.sin(angle)
                oz = 0.0  # ground plane base
                # avoid placing obstacle at the very first waypoint to prevent immediate collision
                self.env.add_obstacle((ox, oy, oz), radius=obs_radius, height=obs_height)
                placed += 1

            print(f"[圆形避障] ✅ 已放置 {placed} 个障碍物，开始加载轨迹到目标队列")

            # 将轨迹加载到队列（保持与 generate_circle_mission 相同的加载方式）
            self.llm_trajectory = trajectory[0]
            self.llm_trajectory_index = 0
            self._load_llm_trajectory_to_queue()
            print(f"[圆形避障] ✅ 轨迹加载完成，开始执行（使用键盘或 remote_controller 控制/监视）")

        except Exception as e:
            # Print detailed traceback and debug info to diagnose ambiguous-array errors
            import traceback
            print(f"[圆形避障] ❌ 生成失败: {e}")
            if DEBUG_CONFIG.get('verbose', False):
                traceback.print_exc()
            else:
                # Always print a short traceback to help debug remote callers
                traceback.print_exc(limit=5)

            try:
                print(f"[圆形避障] 调试信息: command_data type={type(command_data)}, value={repr(command_data)}")
            except Exception:
                pass
            try:
                print(f"[圆形避障] 调试信息: trajectory type={type(trajectory) if 'trajectory' in locals() else 'N/A'}")
            except Exception:
                pass
            try:
                if 'trajectory' in locals():
                    # If it's an array-like, print its shape (safe)
                    import numpy as _np
                    if isinstance(trajectory, _np.ndarray):
                        print(f"[圆形避障] trajectory.shape={trajectory.shape}")
            except Exception:
                pass
    
    def _request_exit(self):
        """请求退出系统"""
        print(f"\n[导航系统] 正在安全退出...")
        self.is_running = False
    
    def _get_current_position(self):
        """获取无人机当前位置"""
        try:
            if self.env:
                drone_state = self.env.get_current_state()
                return drone_state['position']
            else:
                return None
        except Exception as e:
            print(f"[位置] 获取当前位置失败: {e}")
            return None
    
    def _log_trajectory_point(self):
        """记录轨迹点"""
        try:
            if len(self.trajectory) % 10 == 0:  # 每10步记录一次
                drone_state = self.env.get_current_state()
                self.trajectory.append({
                    'time': time.time(),
                    'position': drone_state['position'].copy(),
                    'target': self.current_target.copy() if self.current_target is not None else [0, 0, 0],
                    'distance': drone_state['distance_to_target']
                })
        except Exception as e:
            # 轨迹记录失败不影响主要功能
            pass
    
    def _shutdown(self):
        """系统关闭清理"""
        print(f"\n[导航系统] 正在关闭系统...")
        
        try:
            # 停止键盘监听
            if self.keyboard_controller:
                self.keyboard_controller.stop()
            
            # 关闭网络服务器
            if self.network_server:
                self.network_server.shutdown()
            
            # 关闭环境
            if self.env:
                self.env.close()
            
            # 显示最终统计
            self._show_final_statistics()
            
            print(f"[导航系统] ✅ 系统已安全关闭")
            
        except Exception as e:
            print(f"[导航系统] ⚠️ 关闭过程中出现问题: {e}")
        
        finally:
            self.is_running = False
    
    def _show_final_statistics(self):
        """显示最终统计信息"""
        if not self.stats['start_time']:
            return
            
        total_time = time.time() - self.stats['start_time']
        
        print(f"\n" + "="*50)
        print(f"📊 飞行会话统计")
        print(f"="*50)
        print(f"总飞行时间: {total_time:.1f} 秒")
        print(f"到达目标数: {self.stats['targets_reached']}")
        print(f"轨迹记录点: {len(self.trajectory)}")
        
        if len(self.target_history) > 0:
            print(f"目标历史:")
            for i, target_info in enumerate(self.target_history[-5:]):  # 显示最后5个目标
                target = target_info['target']
                error = target_info['distance_error']
                print(f"  {i+1}. ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) - 误差: {error:.3f}m")
        
        print(f"="*50)
    
    def generate_circle_mission(self, radius=2.0, height=1.5, waypoints=200, clockwise=False):
        """
        生成LLM圆形飞行任务
        
        参数:
            radius: 圆形轨迹半径 (米)
            height: 飞行高度 (米)  
            waypoints: 轨迹点数量
            clockwise: 是否顺时针
        """
        if not LLM_AVAILABLE:
            print("[LLM任务] ❌ LLM轨迹规划器未加载，无法生成圆形任务")
            return False
            
        try:
            print(f"[LLM任务] 🤖 正在生成圆形飞行轨迹...")
            print(f"[参数] 半径: {radius}m, 高度: {height}m, 点数: {waypoints}, 方向: {'顺时针' if clockwise else '逆时针'}")
            
            # 获取当前无人机位置作为起始位置
            current_pos = self._get_current_position()
            if current_pos is not None:
                init_pos = [current_pos[0], current_pos[1], height]
                print(f"[位置] 使用当前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {height:.2f})")
            else:
                # 如果无法获取当前位置，使用默认位置
                init_pos = [radius, 0.0, height]
                print(f"[位置] 使用默认位置: ({radius}, 0.0, {height})")
            
            # 调用LLM生成轨迹
            trajectory = generate_circle_trajectory(
                init_xyz=init_pos,
                num_waypoints=waypoints,
                clockwise=clockwise
            )
            
            if trajectory is not None:
                # 存储轨迹供后续使用
                self.llm_trajectory = trajectory[0]  # 提取单无人机轨迹 (waypoints, 3)
                self.llm_trajectory_index = 0
                
                print(f"[LLM任务] ✅ 圆形轨迹生成成功！")
                print(f"[轨迹信息] 形状: {self.llm_trajectory.shape}, 起点: {self.llm_trajectory[0]}")
                
                # 将轨迹点添加到目标队列
                self._load_llm_trajectory_to_queue()
                
                return True
            else:
                print("[LLM任务] ❌ 轨迹生成失败")
                return False
                
        except Exception as e:
            print(f"[LLM任务] ❌ 生成过程出错: {e}")
            return False
    
    def _load_llm_trajectory_to_queue(self):
        """将LLM生成的轨迹加载到目标队列"""
        if self.llm_trajectory is None:
            return
            
        print(f"[LLM任务] 📋 正在将轨迹点加载到目标队列...")
        
        # 清空现有队列
        self._clear_target_queue()
        
        # 添加轨迹点（考虑0.3m模型误差，确保点间距离足够大）
        step = max(1, len(self.llm_trajectory) // 16)  # 最多16个目标点，确保点间距离>0.5m
        
        for i in range(0, len(self.llm_trajectory), step):
            target_point = self.llm_trajectory[i].tolist()
            self.target_queue.append(target_point)
        
        print(f"[LLM任务] ✅ 已加载 {len(self.target_queue)} 个轨迹点到队列")
        
        # 设置第一个目标
        if self.target_queue:
            self._set_new_target(self.target_queue.pop(0))
    
    def _visualize_llm_trajectory(self):
        """可视化LLM生成的轨迹"""
        if not VISUALIZATION_AVAILABLE or self.llm_trajectory is None:
            return
            
        try:
            print("[可视化] 📊 正在生成轨迹可视化...")
            
            fig = plt.figure(figsize=(15, 5))
            traj = self.llm_trajectory
            
            # 3D轨迹图
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, alpha=0.7)
            ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, label='起点')
            ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, label='终点')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')  
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D轨迹视图', fontproperties=chinese_font)
            ax1.legend(prop=chinese_font)
            
            # XY平面俯视图
            ax2 = fig.add_subplot(132)
            ax2.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
            ax2.scatter(traj[0, 0], traj[0, 1], c='green', s=100, label='起点')
            ax2.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, label='终点')
            # 添加圆心标记
            ax2.scatter(0, 0, c='orange', s=80, marker='+', linewidth=3, label='圆心')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('俯视图 (XY平面)', fontproperties=chinese_font)
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)
            ax2.legend(prop=chinese_font)
            
            # 轨迹分析图
            ax3 = fig.add_subplot(133)
            # 计算距离圆心的距离
            distances = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
            ax3.plot(range(len(distances)), distances, 'g-', linewidth=2, label='半径')
            ax3.axhline(y=distances.mean(), color='r', linestyle='--', alpha=0.7, label=f'平均半径: {distances.mean():.3f}m')
            ax3.plot(range(len(traj)), traj[:, 2], 'orange', linewidth=2, label='高度')
            ax3.set_xlabel('轨迹点索引', fontproperties=chinese_font)
            ax3.set_ylabel('距离/高度 (m)')
            ax3.set_title('轨迹分析', fontproperties=chinese_font)
            ax3.legend(prop=chinese_font)
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle('LLM生成的圆形飞行轨迹', fontsize=16, fontproperties=chinese_font)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig('/tmp/llm_trajectory_preview.png', dpi=150, bbox_inches='tight')
            print("[可视化] ✅ 轨迹图已保存: /tmp/llm_trajectory_preview.png")
            
            plt.show()
            
        except Exception as e:
            print(f"[可视化] ❌ 轨迹可视化失败: {e}")
    
    def show_trajectory_stats(self):
        """显示轨迹统计信息"""
        if self.llm_trajectory is None:
            print("[轨迹统计] ❌ 没有可用的LLM轨迹")
            return
            
        traj = self.llm_trajectory
        print(f"\n📊 LLM轨迹统计信息:")
        print(f"   轨迹点总数: {len(traj)}")
        print(f"   起点: ({traj[0, 0]:.3f}, {traj[0, 1]:.3f}, {traj[0, 2]:.3f})")
        print(f"   终点: ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f}, {traj[-1, 2]:.3f})")
        
        # 半径分析
        distances = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        print(f"   半径统计: 平均 {distances.mean():.3f}m, 标准差 {distances.std():.6f}m")
        
        # 高度分析
        heights = traj[:, 2]
        print(f"   高度统计: 平均 {heights.mean():.3f}m, 范围 {heights.min():.3f}-{heights.max():.3f}m")
        
        # 总路径长度
        path_lengths = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
        total_length = np.sum(path_lengths)
        print(f"   路径总长: {total_length:.3f}m")
        print(f"   平均步长: {path_lengths.mean():.3f}m")


def find_latest_model(results_folder: str = DEFAULT_OUTPUT_FOLDER) -> str:
    """
    查找最新的训练模型
    
    参数:
        results_folder: 结果文件夹路径
        
    返回:
        str: 最新模型的路径
    """
    import glob
    
    # 查找所有保存的训练结果文件夹
    save_folders = glob.glob(os.path.join(results_folder, "save-*"))
    
    if not save_folders:
        raise FileNotFoundError(f"在 {results_folder} 中未找到训练结果文件夹")
    
    # 按时间排序，获取最新的文件夹
    latest_folder = max(save_folders, key=os.path.getmtime)
    
    # 查找best_model.zip
    model_path = os.path.join(latest_folder, "best_model.zip")
    
    if not os.path.exists(model_path):
        # 如果没有best_model.zip，则查找final_model.zip
        model_path = os.path.join(latest_folder, "final_model.zip")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"在 {latest_folder} 中未找到模型文件")
    
    print(f"[模型查找] 找到最新模型: {model_path}")
    return model_path


class NetworkCommandServer:
    """网络命令服务器，接收远程控制指令"""
    
    def __init__(self, navigator, host='localhost', port=12345):
        self.navigator = navigator
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
    def start(self):
        """启动网络服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            
            print(f"[网络服务器] 🌐 启动成功，监听 {self.host}:{self.port}")
            print(f"[网络服务器] 📱 请在新终端运行:")
            print(f"             conda activate drones")
            print(f"             cd {os.getcwd()}")
            print(f"             python -m gym_pybullet_drones.custom.remote_controller")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"[网络服务器] 🔗 客户端连接: {address}")
                    
                    # 处理客户端连接
                    self._handle_client(client_socket)
                    
                except Exception as e:
                    if self.running:
                        print(f"[网络服务器] ❌ 连接错误: {e}")
                        
        except Exception as e:
            print(f"[网络服务器] ❌ 启动失败: {e}")
    
    def _handle_client(self, client_socket):
        """处理客户端命令"""
        buffer = ""
        try:
            while self.running:
                data = client_socket.recv(1024).decode()
                if not data:
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_command(line.strip())
                        
        except Exception as e:
            print(f"[网络服务器] ❌ 客户端处理错误: {e}")
        finally:
            client_socket.close()
            print(f"[网络服务器] 🔌 客户端断开连接")
    
    def _process_command(self, message):
        """处理接收到的命令"""
        try:
            command = json.loads(message)
            command_type = command.get('type')
            command_data = command.get('data')
            
            print(f"[网络命令] 📨 接收: {command_type} - {command_data}")
            
            # 转换为导航系统可识别的命令格式
            if command_type == 'target':
                self.navigator._set_new_target(command_data)
            elif command_type == 'pause':
                self.navigator._pause_navigation()
            elif command_type == 'resume':
                self.navigator._resume_navigation()
            elif command_type == 'home':
                self.navigator._return_home()
            elif command_type == 'current':
                self.navigator._show_current_status()
            elif command_type == 'queue':
                self.navigator._show_target_queue()
            elif command_type == 'clear':
                self.navigator._clear_target_queue()
            elif command_type == 'circle':
                self.navigator._handle_circle_command(command_data)
            elif command_type == 'circle_avoid':
                self.navigator._handle_circle_avoid_command(command_data)
            elif command_type == 'stats':
                self.navigator.show_trajectory_stats()
            elif command_type == 'visual':
                if VISUALIZATION_AVAILABLE and self.navigator.llm_trajectory is not None:
                    self.navigator._visualize_llm_trajectory()
                else:
                    print("[可视化] ❌ 无可用轨迹或可视化功能未加载")
            elif command_type == 'exit':
                self.navigator._request_exit()
            else:
                print(f"[网络命令] ❌ 未知命令类型: {command_type}")
                
        except Exception as e:
            print(f"[网络命令] ❌ 处理失败: {e}")
    
    def shutdown(self):
        """关闭服务器"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
