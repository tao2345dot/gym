"""
SAC 导航模型测试脚本

用于测试训练好的导航模型（15维观测，包含目标位置）

使用方法：
    python -m gym_pybullet_drones.custom.start_sac_navigation \\
        --model_path results/sac-navigation-XX.XX.XXXX_XX.XX.XX/best_model.zip \\
        --gui true
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import numpy as np
from stable_baselines3 import SAC

from gym_pybullet_drones.envs.NavigationAviary import NavigationAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import str2bool


def test_navigation_model(model_path, gui=True, num_episodes=5):
    """
    测试导航模型
    
    参数：
        model_path: 模型路径
        gui: 是否显示 GUI
        num_episodes: 测试轮数
    """
    
    print("=" * 80)
    print("🚁 SAC 导航模型测试")
    print(f"📁 模型路径: {model_path}")
    print(f"🎮 GUI: {'开启' if gui else '关闭'}")
    print(f"📊 测试轮数: {num_episodes}")
    print("=" * 80)
    
    # 加载模型
    print("\n🧠 加载模型...")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return
    
    model = SAC.load(model_path)
    print("  ✅ 模型加载成功")
    
    # 创建测试环境
    print("\n🌍 创建环境...")
    env = NavigationAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=gui
    )
    print("  ✅ 环境创建成功")
    
    # 开始测试
    print("\n" + "=" * 80)
    print("🚀 开始测试...")
    print("=" * 80)
    
    success_count = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        init_pos = env.INIT_XYZS[0]
        target_pos = env.TARGET_POS[0]
        init_dist = np.linalg.norm(target_pos - init_pos)
        
        print(f"\n📍 Episode {episode+1}/{num_episodes}")
        print(f"  起点: [{init_pos[0]:6.2f}, {init_pos[1]:6.2f}, {init_pos[2]:6.2f}]")
        print(f"  目标: [{target_pos[0]:6.2f}, {target_pos[1]:6.2f}, {target_pos[2]:6.2f}]")
        print(f"  距离: {init_dist:.2f}m")
        
        done = False
        step = 0
        min_dist = init_dist
        
        while not done:
            # 模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # 更新最小距离
            current_dist = info['distance_to_target']
            if current_dist < min_dist:
                min_dist = current_dist
            
            # 每秒打印一次状态（30Hz）
            if step % 30 == 0:
                state = env._getDroneStateVector(0)
                pos = state[0:3]
                vel = state[10:13]
                rpy = state[7:10]
                
                print(f"    [{step:3d}] "
                      f"pos=[{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] "
                      f"dist={current_dist:5.3f}m "
                      f"vel={np.linalg.norm(vel):4.2f}m/s "
                      f"tilt={np.rad2deg(np.sqrt(rpy[0]**2 + rpy[1]**2)):4.1f}°")
        
        # Episode 结束
        is_success = info.get('is_success', False)
        if is_success:
            success_count += 1
        
        total_steps += step
        
        print(f"  ✅ 结果: {'✅ 成功到达' if is_success else '❌ 未到达'}")
        print(f"  📏 最近距离: {min_dist:.3f}m")
        print(f"  ⏱️ 总步数: {step} ({step/30:.1f}秒)")
        print(f"  🏆 成功率: {success_count}/{episode+1} = {100*success_count/(episode+1):.1f}%")
    
    # 测试总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print(f"  总轮数: {num_episodes}")
    print(f"  成功数: {success_count}")
    print(f"  成功率: {100*success_count/num_episodes:.1f}%")
    print(f"  平均步数: {total_steps/num_episodes:.1f} ({total_steps/num_episodes/30:.1f}秒)")
    print("=" * 80)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SAC 导航模型测试'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（.zip 文件）')
    parser.add_argument('--gui', type=str2bool, default=True,
                        help='是否显示 GUI（默认: True）')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='测试轮数（默认: 5）')
    
    args = parser.parse_args()
    
    test_navigation_model(
        model_path=args.model_path,
        gui=args.gui,
        num_episodes=args.num_episodes
    )
