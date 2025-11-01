"""
SAC 连续导航快速测试脚本

用于测试 SAC 连续导航系统的基本功能（无需训练模型）
"""

import os
import sys
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass

import time
import numpy as np
from stable_baselines3 import SAC

from gym_pybullet_drones.custom.space_expander import ExtendedHoverAviary
from gym_pybullet_drones.custom.config_continuous import *

def test_sac_continuous_navigation():
    """
    快速测试 SAC 连续导航功能
    使用随机策略（不需要训练好的模型）
    """
    print("\n" + "="*70)
    print("🧪 SAC 连续导航快速测试")
    print("="*70)
    print("说明: 此测试使用随机策略，无需训练模型")
    print("仅用于验证系统基本功能和环境设置")
    print("="*70 + "\n")
    
    # 创建环境
    print("[1/5] 创建扩展环境...")
    init_pos = np.array([DEFAULT_INIT_POS])
    init_rpy = np.array([[0, 0, 0]])
    
    env = ExtendedHoverAviary(
        initial_xyzs=init_pos,
        initial_rpys=init_rpy,
        gui=True,
        record=False,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        target_pos=DEFAULT_TARGET_POS,
        obstacles=True
    )
    print("✅ 环境创建成功")
    
    # 添加柱子可视化（固定大小，随机位置）
    print("\n[2/5] 添加柱子...")
    
    # 生成 4 根随机位置的柱子
    obstacles = []
    pillar_radius = 0.08  # 与蓝色路径点大小一致
    pillar_height = 2.0
    
    import random
    random.seed(42)  # 使用固定种子保证可复现
    
    for i in range(4):
        # 随机生成位置（避免太靠近原点）
        x = random.uniform(-1.5, 1.5)
        y = random.uniform(-1.5, 1.5)
        
        # 确保不在原点附近
        while (x**2 + y**2) < 0.5**2:
            x = random.uniform(-1.5, 1.5)
            y = random.uniform(-1.5, 1.5)
        
        obstacles.append({
            'pos': [x, y, pillar_height / 2],
            'radius': pillar_radius,
            'height': pillar_height
        })
    
    import pybullet as p
    for i, obs in enumerate(obstacles):
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=obs['radius'],
            length=obs['height'],
            rgbaColor=[0.6, 0.6, 0.6, 1.0],  # 灰色柱子
            physicsClientId=env.CLIENT
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=obs['radius'],
            height=obs['height'],
            physicsClientId=env.CLIENT
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=obs['pos'],
            physicsClientId=env.CLIENT
        )
        print(f"  柱子 {i+1}: 位置=({obs['pos'][0]:.2f}, {obs['pos'][1]:.2f}), 半径={obs['radius']:.2f}m, 高度={obs['height']:.2f}m")
    print("✅ 已添加 4 根柱子")
    
    # 添加目标标记
    print("\n[3/5] 添加目标标记...")
    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.1,
        rgbaColor=[0.2, 0.8, 0.2, 0.8],
        physicsClientId=env.CLIENT
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=DEFAULT_TARGET_POS,
        physicsClientId=env.CLIENT
    )
    print(f"✅ 目标位置: {DEFAULT_TARGET_POS}")
    
    # 重置环境
    print("\n[4/5] 重置环境...")
    obs, info = env.reset(seed=42, options={})
    print(f"✅ 初始观测维度: {obs.shape}")
    
    # 运行测试
    print("\n[5/5] 开始测试导航...")
    print("\n按 Ctrl+C 停止测试\n")
    
    trajectory = []
    targets = [
        DEFAULT_TARGET_POS,
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 0.5],
        [0.0, 0.0, 1.5]
    ]
    current_target_idx = 0
    
    try:
        for step in range(500):  # 运行 500 步
            start_time = time.time()
            
            # 获取当前位置（确保为 numpy 数组以便后续计算/格式化）
            current_pos = np.array(env.pos[0])
            trajectory.append(current_pos.copy())
            
            # 检查是否到达当前目标
            if current_target_idx < len(targets):
                target = np.array(targets[current_target_idx])
                distance = np.linalg.norm(current_pos - target)
                
                if distance < 0.2:
                    print(f"✅ 到达目标 {current_target_idx + 1}/{len(targets)}: {target}")
                    current_target_idx += 1
                    if current_target_idx < len(targets):
                        env.TARGET_POS = targets[current_target_idx]
                        print(f"📍 新目标: {targets[current_target_idx]}")
            
            # 使用随机动作（实际使用中应该用 SAC 模型）
            action = env.action_space.sample()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 显示进度
            if step % 50 == 0:
                # 确保 target 也是 numpy 数组以支持 round()
                target_arr = np.array(targets[min(current_target_idx, len(targets)-1)])
                print(f"Step {step}: Pos={current_pos.round(2)}, "
                      f"Target={target_arr.round(2)}, "
                      f"Dist={distance:.2f}m")
            
            # 渲染
            env.render()
            
            # 控制帧率
            elapsed = time.time() - start_time
            if elapsed < env.CTRL_TIMESTEP:
                time.sleep(env.CTRL_TIMESTEP - elapsed)
    
    except KeyboardInterrupt:
        print("\n[测试] 接收到中断信号")
    
    finally:
        # 清理
        print("\n[清理] 关闭环境...")
        env.close()
        
        # 统计
        print("\n" + "="*70)
        print("📊 测试统计")
        print("="*70)
        print(f"运行步数: {len(trajectory)}")
        print(f"到达目标数: {current_target_idx}/{len(targets)}")
        if len(trajectory) > 0:
            trajectory_array = np.array(trajectory)
            total_distance = np.sum(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))
            print(f"飞行距离: {total_distance:.2f} m")
        print("="*70)
        
        print("\n✅ 测试完成")
        print("\n💡 提示:")
        print("  1. 此测试使用随机策略，性能有限")
        print("  2. 实际使用请先训练 SAC 模型:")
        print("     python -m gym_pybullet_drones.custom.sac_learn --local true --gui false")
        print("  3. 然后运行完整连续导航:")
        print("     python -m gym_pybullet_drones.custom.start_sac_continuous --model_path <model.zip>")


if __name__ == '__main__':
    test_sac_continuous_navigation()
