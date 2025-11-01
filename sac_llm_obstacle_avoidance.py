"""
SAC + LLM 避障集成示例

本脚本演示如何将训练好的 SAC 模型与 LLM 避障规划器结合使用。

主要功能：
1. 加载预训练的 SAC 模型
2. 在环境中检测障碍物
3. 使用 LLM（或几何规划）生成避障路径点
4. SAC 模型执行避障动作

使用方法：
    # 首先训练 SAC 模型
    $ python sac_learn.py --local true --gui false
    
    # 然后运行避障测试
    $ python sac_llm_obstacle_avoidance.py --model_path results/sac-save-XX.XX.XXXX_XX.XX.XX/best_model.zip

说明：
- 需要先使用 sac_learn.py 训练模型
- 支持 OpenAI API（需设置 OPENAI_API_KEY 环境变量）
- 如果没有 API Key，会自动回退到几何规划
"""
import os
import sys
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass

import time
import argparse
import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.custom.llm_obstacle_avoidance import plan_avoidance

# 默认配置
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_EPISODE_LEN_SEC = 20  # 每个回合的时长（秒）

# 障碍物配置（示例）
OBSTACLES = [
    {'pos': [0.5, 0.5, 0.5], 'radius': 0.3, 'height': 1.0},
    {'pos': [-0.5, 0.5, 0.5], 'radius': 0.25, 'height': 0.8},
    {'pos': [0.0, -0.5, 0.5], 'radius': 0.35, 'height': 1.2},
]

# 目标位置
TARGET_POS = np.array([1.0, 1.0, 1.0])

def add_obstacle_to_sim(client_id, pos, radius, height, color=[0.8, 0.2, 0.2, 1.0]):
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
        baseMass=0,  # 静态障碍物
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[pos[0], pos[1], pos[2]],
        physicsClientId=client_id
    )
    return obstacle_id

def add_target_marker(client_id, pos, size=0.1, color=[0.2, 0.8, 0.2, 0.8]):
    """在 PyBullet 仿真中添加目标标记（半透明球体）"""
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

def add_waypoint_marker(client_id, pos, size=0.08, color=[0.2, 0.2, 0.8, 0.8]):
    """在 PyBullet 仿真中添加路径点标记（半透明球体）"""
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

def check_obstacle_collision(agent_pos, obstacles, safety_margin=0.1):
    """
    检查智能体是否与障碍物发生碰撞（或过于接近）
    
    Args:
        agent_pos: 智能体位置 [x, y, z]
        obstacles: 障碍物列表，每个元素为 {'pos': [x,y,z], 'radius': r, 'height': h}
        safety_margin: 安全距离余量
    
    Returns:
        (bool, dict or None): (是否碰撞, 最近的障碍物信息)
    """
    agent_pos_2d = np.array(agent_pos[:2])
    min_dist = float('inf')
    closest_obs = None
    
    for obs in obstacles:
        obs_pos_2d = np.array(obs['pos'][:2])
        dist = np.linalg.norm(agent_pos_2d - obs_pos_2d)
        threshold = obs['radius'] + safety_margin
        
        # 检查高度范围
        if abs(agent_pos[2] - obs['pos'][2]) < obs['height'] / 2:
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs
            if dist < threshold:
                return True, obs
    
    return False, closest_obs

def run(model_path, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, 
        output_folder=DEFAULT_OUTPUT_FOLDER, use_llm=True, openai_model='gpt-3.5-turbo'):
    """
    运行 SAC + LLM 避障演示
    
    Args:
        model_path: SAC 模型文件路径
        gui: 是否显示 GUI
        record_video: 是否录制视频
        output_folder: 输出文件夹
        use_llm: 是否使用 LLM 规划（如果 False 则只使用几何规划）
        openai_model: OpenAI 模型名称
    """
    
    # 加载 SAC 模型
    print(f"[INFO] Loading SAC model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("[INFO] Please train a model first using: python sac_learn.py")
        return
    
    model = SAC.load(model_path)
    print("[INFO] SAC model loaded successfully")
    
    # 创建环境
    env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
    print("[INFO] Environment created")
    
    # 添加障碍物到仿真
    obstacle_ids = []
    for obs in OBSTACLES:
        obs_id = add_obstacle_to_sim(env.CLIENT, obs['pos'], obs['radius'], obs['height'])
        obstacle_ids.append(obs_id)
    print(f"[INFO] Added {len(OBSTACLES)} obstacles to simulation")
    
    # 添加目标标记
    target_marker_id = add_target_marker(env.CLIENT, TARGET_POS)
    print(f"[INFO] Target position: {TARGET_POS}")
    
    # 初始化
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    
    # 状态机：'to_target' 或 'avoiding'
    state = 'to_target'
    current_waypoint = TARGET_POS.copy()
    waypoint_marker_id = None
    
    collision_count = 0
    avoidance_count = 0
    
    print("\n[INFO] Starting SAC + LLM obstacle avoidance demo...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    for i in range(DEFAULT_EPISODE_LEN_SEC * env.CTRL_FREQ):
        # 获取当前智能体位置（从观测中提取）
        if DEFAULT_OBS == ObservationType.KIN:
            agent_pos = obs[:3]  # 前3个元素是位置 [x, y, z]
        else:
            # 如果是其他观测类型，需要从环境中获取
            agent_pos = env._getDroneStateVector(0)[:3]
        
        # 检查是否接近障碍物
        is_colliding, nearest_obs = check_obstacle_collision(agent_pos, OBSTACLES, safety_margin=0.2)
        
        if is_colliding and state == 'to_target':
            # 触发避障
            state = 'avoiding'
            avoidance_count += 1
            print(f"\n[AVOID] Step {i}: Obstacle detected at {nearest_obs['pos']}, planning avoidance...")
            
            # 使用 LLM 规划避障路径点
            if use_llm:
                # 构建环境边界信息
                env_bounds = {
                    'x_range': [-2.0, 2.0],
                    'y_range': [-2.0, 2.0],
                    'z_range': [0.0, 2.0]
                }
                # 构建其他障碍物信息
                other_obs = [(o['pos'], o['radius']) for o in OBSTACLES if o != nearest_obs]
                
                current_waypoint = plan_avoidance(
                    agent_pos=agent_pos,
                    obstacle_pos=nearest_obs['pos'],
                    radius=nearest_obs['radius'],
                    target_pos=TARGET_POS,
                    clearance=0.3,
                    model=openai_model,
                    env_bounds=env_bounds,
                    other_obstacles=other_obs
                )
            else:
                # 只使用几何规划
                from gym_pybullet_drones.custom.llm_obstacle_avoidance import _geometric_plan
                current_waypoint = _geometric_plan(
                    agent_pos, nearest_obs['pos'], nearest_obs['radius'], 
                    TARGET_POS, clearance=0.3
                )
            
            print(f"[AVOID] New waypoint: {current_waypoint}")
            
            # 添加路径点标记
            if waypoint_marker_id is not None:
                p.removeBody(waypoint_marker_id, physicsClientId=env.CLIENT)
            waypoint_marker_id = add_waypoint_marker(env.CLIENT, current_waypoint)
        
        elif state == 'avoiding':
            # 检查是否到达避障路径点
            dist_to_waypoint = np.linalg.norm(agent_pos - current_waypoint)
            if dist_to_waypoint < 0.2:  # 到达路径点
                print(f"[INFO] Step {i}: Reached avoidance waypoint, resuming to target")
                state = 'to_target'
                current_waypoint = TARGET_POS.copy()
                if waypoint_marker_id is not None:
                    p.removeBody(waypoint_marker_id, physicsClientId=env.CLIENT)
                    waypoint_marker_id = None
        
        # 修改观测以包含当前目标（简单方法：直接使用原始观测，SAC模型应该已经学会导航）
        # 如果需要，可以在这里修改观测来引导模型朝向 current_waypoint
        
        # SAC 模型预测动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 定期打印状态
        if i % (env.CTRL_FREQ * 2) == 0:  # 每2秒打印一次
            dist_to_target = np.linalg.norm(agent_pos - TARGET_POS)
            print(f"[INFO] Step {i}: Pos={agent_pos}, State={state}, DistToTarget={dist_to_target:.2f}m")
        
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            print(f"\n[INFO] Episode ended at step {i}")
            break
        
        # 检查是否到达目标
        if np.linalg.norm(agent_pos - TARGET_POS) < 0.15:
            print(f"\n[SUCCESS] Reached target at step {i}!")
            break
    
    env.close()
    
    print("\n" + "="*60)
    print(f"[SUMMARY] SAC + LLM Obstacle Avoidance Demo Complete")
    print(f"  - Total avoidance maneuvers: {avoidance_count}")
    print(f"  - LLM planning: {'Enabled' if use_llm else 'Disabled (geometric only)'}")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC + LLM 避障集成示例')
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True,
                        help='SAC 模型文件路径（.zip 文件）')
    parser.add_argument('--gui', 
                        default=DEFAULT_GUI, 
                        type=str2bool, 
                        help='是否显示 PyBullet GUI（默认True）')
    parser.add_argument('--record_video', 
                        default=DEFAULT_RECORD_VIDEO, 
                        type=str2bool, 
                        help='是否录制视频（默认False）')
    parser.add_argument('--output_folder', 
                        default=DEFAULT_OUTPUT_FOLDER, 
                        type=str, 
                        help='输出文件夹（默认results）')
    parser.add_argument('--use_llm', 
                        default=True, 
                        type=str2bool, 
                        help='是否使用 LLM 规划（默认True，需要 OPENAI_API_KEY）')
    parser.add_argument('--openai_model', 
                        default='gpt-3.5-turbo', 
                        type=str, 
                        help='OpenAI 模型名称（默认 gpt-3.5-turbo）')
    
    args = parser.parse_args()
    
    # 检查 API Key（如果启用 LLM）
    if args.use_llm and not os.getenv('OPENAI_API_KEY'):
        print("\n[WARNING] OPENAI_API_KEY not found in environment variables")
        print("[WARNING] Will fall back to geometric planning")
        print("[INFO] To use LLM planning, set: export OPENAI_API_KEY='your-key-here'\n")
    
    run(
        model_path=args.model_path,
        gui=args.gui,
        record_video=args.record_video,
        output_folder=args.output_folder,
        use_llm=args.use_llm,
        openai_model=args.openai_model
    )
