"""
SAC 改进训练脚本 - 优化版本

改进内容：
1. ✅ 增加训练步数：从 1M → 3M
2. ✅ 调整奖励函数：更简单的目标和更好的奖励设计
3. ✅ 使用更简单的目标：降低高度从 1.0m → 0.3m
4. ✅ 优化超参数：更适合小空间导航
5. ✅ 改进学习策略：更快的收敛

使用方法：
    python -m gym_pybullet_drones.custom.sac_learn_improved --gui false
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
# 改进配置
# ============================================================================

# 环境配置
NUM_ENV = 8  # 减少并行环境数量以提高稳定性
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

# 改进的训练参数
TRAINING_CONFIG = {
    'total_timesteps': int(3e6),      # 增加到 3M 步
    'buffer_size': 500000,             # 适中的 replay buffer
    'learning_starts': 5000,           # 更快开始学习
    'batch_size': 256,
    'learning_rate': 1e-4,             # 较小的学习率以提高稳定性
    'tau': 0.01,                       # 更快的目标网络更新
    'gamma': 0.98,                     # 稍微降低折扣因子
    'train_freq': (1, 'step'),
    'gradient_steps': 1,
    'ent_coef': 'auto',                # 自动调整熵系数
    'target_entropy': 'auto',
}


class ImprovedHoverAviary(gym.Env):
    """
    改进的悬停环境
    
    改进点：
    1. 更简单的目标：固定在 [0, 0, 0.3]
    2. 更好的奖励函数：距离 + 速度 + 稳定性
    3. 更宽松的成功条件
    """
    
    def __init__(self, gui=False, record=False, obs=ObservationType.KIN, act=ActionType.RPM):
        from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
        
        # 使用基础环境但修改目标位置
        self.base_env = HoverAviary(gui=gui, record=record, obs=obs, act=act)
        
        # 设置更简单的目标
        self.target_pos = np.array([0.0, 0.0, 0.3])  # 降低高度
        self.base_env.TARGET_POS = self.target_pos
        
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
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        self.episode_step = 0
        self.episode_reward = 0
        
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # 确保目标位置正确
        self.base_env.TARGET_POS = self.target_pos
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        # 执行基础环境的 step
        obs, base_reward, terminated, truncated, info = self.base_env.step(action)
        
        # 计算改进的奖励
        reward = self._compute_improved_reward(obs)
        
        self.episode_step += 1
        self.episode_reward += reward
        
        # 检查是否成功到达
        current_pos = self.base_env.pos[0]
        distance = np.linalg.norm(current_pos - self.target_pos)
        
        if distance < 0.1:  # 更宽松的成功条件
            self.success_count += 1
            reward += 50.0  # 成功奖励
            info['is_success'] = True
        else:
            info['is_success'] = False
        
        return obs, reward, terminated, truncated, info
    
    def _compute_improved_reward(self, obs):
        """
        改进的奖励函数
        
        组成部分：
        1. 距离奖励（主要）：越近越好
        2. 速度惩罚：速度过大会被惩罚
        3. 稳定性奖励：保持稳定会获得奖励
        """
        current_pos = self.base_env.pos[0]
        current_vel = obs[10:13] if len(obs) > 13 else np.zeros(3)
        
        # 1. 距离奖励（使用指数函数，越近奖励越高）
        distance = np.linalg.norm(current_pos - self.target_pos)
        distance_reward = 10.0 * np.exp(-distance * 3.0)  # 指数衰减
        
        # 2. 到达奖励（距离很近时额外奖励）
        if distance < 0.1:
            distance_reward += 20.0
        elif distance < 0.2:
            distance_reward += 10.0
        
        # 3. 速度惩罚（速度不应过大）
        velocity_magnitude = np.linalg.norm(current_vel)
        velocity_penalty = -0.5 * velocity_magnitude
        
        # 4. 稳定性奖励（速度很小时奖励）
        if velocity_magnitude < 0.1:
            stability_reward = 2.0
        else:
            stability_reward = 0.0
        
        # 5. 高度惩罚（如果飞得太高）
        if current_pos[2] > 0.5:
            height_penalty = -5.0 * (current_pos[2] - 0.5)
        else:
            height_penalty = 0.0
        
        # 6. 边界惩罚（不要飞出范围）
        boundary_penalty = 0.0
        if abs(current_pos[0]) > 1.0 or abs(current_pos[1]) > 1.0:
            boundary_penalty = -10.0
        
        # 总奖励
        total_reward = (
            distance_reward + 
            velocity_penalty + 
            stability_reward + 
            height_penalty + 
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
    """
    进度回调：每隔一定步数输出训练进度
    """
    def __init__(self, check_freq: int = 10000, verbose: int = 1):
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
    主训练流程
    """
    
    print("\n" + "="*70)
    print("  🚀 SAC 改进训练脚本")
    print("="*70)
    print(f"  训练步数: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  并行环境: {NUM_ENV}")
    print(f"  目标位置: [0, 0, 0.3] (简化目标)")
    print(f"  改进奖励: 距离 + 速度 + 稳定性")
    print("="*70 + "\n")
    
    # 创建保存目录
    filename = os.path.join(
        output_folder, 
        'sac-improved-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    )
    os.makedirs(filename, exist_ok=True)
    
    print(f"📁 保存路径: {filename}\n")
    
    # 创建训练环境
    print("🏗️  创建训练环境...")
    train_env = make_vec_env(
        ImprovedHoverAviary,
        env_kwargs=dict(gui=False, record=False),
        n_envs=NUM_ENV,
        seed=0
    )
    
    # 创建评估环境
    eval_env = Monitor(ImprovedHoverAviary(gui=False, record=False))
    
    print(f"   ✅ {NUM_ENV} 个并行训练环境已创建")
    print(f"   Action space: {train_env.action_space}")
    print(f"   Observation space: {train_env.observation_space}\n")
    
    # 网络架构（更大的网络）
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256])  # 更大的网络
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
    
    # 目标奖励（根据新的奖励函数调整）
    target_reward = 100.0
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
        eval_freq=int(10000),  # 每 10k 步评估一次
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 进度回调
    progress_callback = ProgressCallback(check_freq=50000)
    
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
    test_env = ImprovedHoverAviary(gui=gui, record=record_video)
    test_env_nogui = ImprovedHoverAviary(gui=False, record=False)
    
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
        description='SAC 改进训练脚本 - 增加步数、优化奖励、简化目标'
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
