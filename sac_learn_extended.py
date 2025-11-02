"""
SAC 扩展训练脚本 - 方案 B：使用 ExtendedHoverAviary

改进内容：
1. ✅ 使用更大的空间：X/Y ∈ [-3, 3], Z ∈ [0.05, 2.5]
2. ✅ 动态目标更新：通过 update_target() 方法
3. ✅ 安全边界检查：避免飞出边界
4. ✅ 禁用截断模式：更适合连续导航
5. ✅ 随机初始目标：每次 episode 不同

使用方法：
    python -m gym_pybullet_drones.custom.sac_learn_extended --gui false
"""

import os
import sys
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass

import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


# ============================================================================
# 方案 B 配置
# ============================================================================

# 环境配置
NUM_ENV = 8
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

# 训练参数（针对更大空间优化）
TRAINING_CONFIG = {
    'total_timesteps': int(5e6),      # 更大空间需要更多训练
    'buffer_size': 1000000,            # 更大的 replay buffer
    'learning_starts': 10000,          # 更多探索
    'batch_size': 256,
    'learning_rate': 3e-4,             # 标准学习率
    'tau': 0.005,                      # 软更新
    'gamma': 0.99,                     # 标准折扣因子
    'train_freq': (1, 'step'),
    'gradient_steps': 1,
    'ent_coef': 'auto',
    'target_entropy': 'auto',
}


class ExtendedTrainingWrapper(gym.Env):
    """
    扩展空间训练包装器
    
    使用 ExtendedHoverAviary 作为基础环境
    添加改进的奖励函数和随机目标生成
    """
    
    def __init__(self, gui=False, record=False):
        from gym_pybullet_drones.custom.space_expander import ExtendedHoverAviary
        
        # 使用扩展空间环境
        self.base_env = ExtendedHoverAviary(
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            initial_xyzs=np.array([[0, 0, 1]]),  # 固定起点
            testing_space={
                'x_range': [-3.0, 3.0],
                'y_range': [-3.0, 3.0],
                'z_range': [0.05, 2.5],
                'disable_truncation': True,  # 禁用截断以便连续导航
                'tilt_limit': 1.5
            }
        )
        
        # 继承空间定义
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # 环境参数
        self.CTRL_FREQ = self.base_env.CTRL_FREQ
        self.EPISODE_LEN_SEC = self.base_env.EPISODE_LEN_SEC
        self.CTRL_TIMESTEP = self.base_env.CTRL_TIMESTEP
        
        # 统计信息
        self.episode_step = 0
        self.episode_reward = 0
        self.success_count = 0
        self.current_target = None
        
    def reset(self, seed=None, options=None):
        """重置环境并生成随机目标"""
        self.episode_step = 0
        self.episode_reward = 0
        
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # ✅ 方案 B：每次生成大范围随机目标
        self.current_target = np.array([
            np.random.uniform(-2.5, 2.5),   # x: -2.5 到 2.5
            np.random.uniform(-2.5, 2.5),   # y: -2.5 到 2.5
            np.random.uniform(0.3, 2.0)     # z: 0.3 到 2.0
        ])
        
        # 更新环境目标
        self.base_env.update_target(self.current_target)
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        # 执行基础环境的 step
        obs, base_reward, terminated, truncated, info = self.base_env.step(action)
        
        # 计算改进的奖励
        reward = self._compute_extended_reward(obs)
        
        self.episode_step += 1
        self.episode_reward += reward
        
        # 检查是否成功到达
        current_pos = self.base_env.pos[0]
        distance = np.linalg.norm(current_pos - self.current_target)
        
        if distance < 0.15:  # 更宽松的成功条件（空间更大）
            self.success_count += 1
            reward += 100.0  # 更大的成功奖励
            info['is_success'] = True
        else:
            info['is_success'] = False
        
        return obs, reward, terminated, truncated, info
    
    def _compute_extended_reward(self, obs):
        """
        扩展空间的奖励函数
        
        针对更大空间和更远距离优化
        """
        current_pos = self.base_env.pos[0]
        current_vel = obs[10:13] if len(obs) > 13 else np.zeros(3)
        
        # 1. 距离奖励（针对大空间调整）
        distance = np.linalg.norm(current_pos - self.current_target)
        
        # 使用分段奖励
        if distance < 0.15:
            distance_reward = 50.0  # 非常接近
        elif distance < 0.5:
            distance_reward = 30.0 * np.exp(-distance)  # 接近
        elif distance < 2.0:
            distance_reward = 20.0 * np.exp(-distance * 0.5)  # 中距离
        else:
            distance_reward = 10.0 * np.exp(-distance * 0.2)  # 远距离
        
        # 2. 前进奖励（鼓励向目标移动）
        direction_to_target = self.current_target - current_pos
        distance_to_target = np.linalg.norm(direction_to_target)
        
        if distance_to_target > 0.01:
            direction_normalized = direction_to_target / distance_to_target
            velocity_towards_target = np.dot(current_vel, direction_normalized)
            progress_reward = max(0, velocity_towards_target * 5.0)  # 向目标移动时奖励
        else:
            progress_reward = 0.0
        
        # 3. 速度惩罚（不要太快）
        velocity_magnitude = np.linalg.norm(current_vel)
        if velocity_magnitude > 2.0:
            velocity_penalty = -1.0 * (velocity_magnitude - 2.0)
        else:
            velocity_penalty = 0.0
        
        # 4. 稳定性奖励（到达后保持稳定）
        if distance < 0.3 and velocity_magnitude < 0.2:
            stability_reward = 10.0
        else:
            stability_reward = 0.0
        
        # 5. 边界惩罚（避免飞出范围）
        boundary_penalty = 0.0
        if abs(current_pos[0]) > 2.8 or abs(current_pos[1]) > 2.8:
            boundary_penalty = -20.0
        if current_pos[2] < 0.1 or current_pos[2] > 2.4:
            boundary_penalty = -20.0
        
        # 总奖励
        total_reward = (
            distance_reward +
            progress_reward +
            velocity_penalty +
            stability_reward +
            boundary_penalty
        )
        
        return total_reward
    
    def render(self):
        """渲染"""
        return self.base_env.render()
    
    def close(self):
        """关闭环境"""
        return self.base_env.close()


class ProgressCallback(BaseCallback):
    """进度回调"""
    def __init__(self, check_freq: int = 20000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"\n{'='*60}")
            print(f"📊 训练进度: {self.num_timesteps:,} / {TRAINING_CONFIG['total_timesteps']:,} 步")
            print(f"   进度: {self.num_timesteps / TRAINING_CONFIG['total_timesteps'] * 100:.1f}%")
            print(f"{'='*60}\n")
        return True


def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, 
        record_video=DEFAULT_RECORD_VIDEO, local=True):
    """
    主训练流程 - 方案 B
    """
    
    print("\n" + "="*70)
    print("  🚀 SAC 扩展训练脚本 - 方案 B：ExtendedHoverAviary")
    print("="*70)
    print(f"  训练步数: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  并行环境: {NUM_ENV}")
    print(f"  空间范围: X/Y ∈ [-3, 3], Z ∈ [0.05, 2.5]")
    print(f"  目标位置: 随机 X/Y ∈ [-2.5, 2.5], Z ∈ [0.3, 2.0]")
    print(f"  改进奖励: 距离 + 前进 + 速度 + 稳定性")
    print(f"  禁用截断: True")
    print("="*70 + "\n")
    
    # 创建保存目录
    filename = os.path.join(
        output_folder, 
        'sac-extended-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    )
    os.makedirs(filename, exist_ok=True)
    
    print(f"📁 保存路径: {filename}\n")
    
    # 创建训练环境
    print("🏗️  创建扩展训练环境...")
    train_env = make_vec_env(
        ExtendedTrainingWrapper,
        env_kwargs=dict(gui=False, record=False),
        n_envs=NUM_ENV,
        seed=0
    )
    
    # 创建评估环境
    eval_env = Monitor(ExtendedTrainingWrapper(gui=False, record=False))
    
    print(f"   ✅ {NUM_ENV} 个并行训练环境已创建")
    print(f"   Action space: {train_env.action_space}")
    print(f"   Observation space: {train_env.observation_space}\n")
    
    # 网络架构（更大的网络以应对更复杂的空间）
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256])
    )
    
    # 创建 SAC 模型
    print("🧠 创建 SAC 模型...")
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        buffer_size=TRAINING_CONFIG['buffer_size'],
        learning_starts=TRAINING_CONFIG['learning_starts'],
        batch_size=TRAINING_CONFIG['batch_size'],
        tau=TRAINING_CONFIG['tau'],
        gamma=TRAINING_CONFIG['gamma'],
        train_freq=TRAINING_CONFIG['train_freq'],
        gradient_steps=TRAINING_CONFIG['gradient_steps'],
        ent_coef=TRAINING_CONFIG['ent_coef'],
        target_entropy=TRAINING_CONFIG['target_entropy'],
        tensorboard_log=filename + '/tb/',
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"   ✅ 模型已创建")
    print(f"   设备: {model.device}\n")
    
    # 回调函数
    print("📋 设置训练回调...")
    
    # 目标奖励（更大空间需要更高阈值）
    target_reward = 500.0
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1
    )
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename + '/',
        log_path=filename + '/',
        eval_freq=int(20000),  # 每 20k 步评估一次
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 进度回调
    progress_callback = ProgressCallback(check_freq=100000)
    
    callbacks = [eval_callback, progress_callback]
    
    print("   ✅ 回调已设置\n")
    
    # 开始训练
    print("🎯 开始训练...\n")
    print("="*70)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=TRAINING_CONFIG['total_timesteps'] if local else 1000,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"✅ 训练完成！")
    print(f"   总时长: {training_time / 60:.1f} 分钟")
    print("="*70 + "\n")
    
    # 保存最终模型
    model.save(filename + '/final_model.zip')
    print(f"💾 模型已保存到: {filename}\n")
    
    # 评估结果
    if os.path.exists(filename + '/evaluations.npz'):
        print("📊 训练过程评估结果:")
        print("-"*70)
        with np.load(filename + '/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(f"  步数 {data['timesteps'][j]:>8,}: "
                      f"平均奖励 {data['results'][j][0]:>8.2f}")
        print("-"*70 + "\n")
    
    # 加载最佳模型进行测试
    print("🧪 测试最佳模型...\n")
    
    if os.path.isfile(filename + '/best_model.zip'):
        model = SAC.load(filename + '/best_model.zip')
        print("   ✅ 已加载最佳模型")
    else:
        print("   ⚠️  未找到最佳模型，使用最终模型")
    
    # 测试环境
    test_env = ExtendedTrainingWrapper(gui=gui, record=record_video)
    test_env_nogui = ExtendedTrainingWrapper(gui=False, record=False)
    
    # 评估
    print("\n📈 评估模型性能...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        test_env_nogui,
        n_eval_episodes=20,
        deterministic=True
    )
    
    print(f"\n{'='*70}")
    print(f"  📊 最终评估结果")
    print(f"{'='*70}")
    print(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*70}\n")
    
    # GUI 测试（如果启用）
    if gui:
        print("🎮 运行可视化测试...\n")
        obs, info = test_env.reset(seed=42)
        episode_reward = 0
        
        for i in range(test_env.EPISODE_LEN_SEC * test_env.CTRL_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            
            test_env.render()
            
            if terminated or truncated:
                print(f"   Episode 回合奖励: {episode_reward:.2f}")
                break
        
        test_env.close()
    
    print("\n✅ 所有任务完成！\n")
    print(f"📁 结果保存在: {filename}")
    print(f"📊 Tensorboard 日志: tensorboard --logdir {filename}/tb/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAC 扩展训练脚本 - 方案 B：使用 ExtendedHoverAviary'
    )
    parser.add_argument(
        '--gui',
        default=DEFAULT_GUI,
        type=str2bool,
        help='是否显示 GUI (默认: False)',
        metavar=''
    )
    parser.add_argument(
        '--record_video',
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help='是否录制视频 (默认: False)',
        metavar=''
    )
    parser.add_argument(
        '--output_folder',
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='输出文件夹 (默认: results)',
        metavar=''
    )
    parser.add_argument(
        '--local',
        default=True,
        type=str2bool,
        help='本地完整训练 (默认: True)',
        metavar=''
    )
    
    args = parser.parse_args()
    
    try:
        run(**vars(args))
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
