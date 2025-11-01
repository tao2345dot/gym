"""
本脚本演示了如何使用 `gym_pybullet_drones` 的 Gymnasium 接口进行强化学习，基于 SAC 算法。

主要功能：
1. 支持单智能体和多智能体环境（HoverAviary / MultiHoverAviary），用于 SAC 算法训练。
2. 集成 stable-baselines3 强化学习库，实现训练、评估、保存和测试。

使用方法：
    $ python sac_learn.py --multiagent false  # 单智能体
    $ python sac_learn.py --multiagent true   # 多智能体

说明：
这是一个基于 SAC (Soft Actor-Critic) 算法的 gym-pybullet-drones 与 stable-baselines3 集成示例。
SAC 是一种 off-policy 算法，适合连续动作空间，通常比 PPO 更加样本高效。
"""
import os
import sys
# 添加父目录到 sys.path，便于导入 gym_pybullet_drones 包
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
# 导入 stable-baselines3 相关模块（SAC 算法）
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# SAC 配置参数
NUM_ENV = 16  # SAC 通常使用较少的并行环境（off-policy 算法更样本高效）
EXP_NAME = "sac_hover_experiment"
CUDA = "cuda:0"

DEFAULT_GUI = True  # 是否显示PyBullet GUI
DEFAULT_RECORD_VIDEO = False  # 是否录制视频
DEFAULT_OUTPUT_FOLDER = 'results'  # 日志/模型保存路径
DEFAULT_COLAB = False  # 是否在Colab环境运行

DEFAULT_OBS = ObservationType('kin') # 观测类型：'kin'（动力学）或 'rgb'（图像）
DEFAULT_ACT = ActionType('rpm') # 动作类型：'rpm'/'pid'/'vel'/'one_d_rpm'/'one_d_pid'
DEFAULT_AGENTS = 2  # 多智能体时的无人机数量
DEFAULT_MA = False  # 是否多智能体

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):
    """
    主流程函数：训练、评估、保存、测试强化学习模型（基于 SAC 算法）。
    参数说明：
        multiagent: 是否多智能体
        output_folder: 日志/模型保存路径
        gui: 是否显示PyBullet GUI
        plot: 是否绘制训练结果
        colab: 是否在Colab环境运行
        record_video: 是否录制视频
        local: 是否本地长时间训练（否则快速测试）
    """

    # 生成保存路径（带时间戳）
    filename = os.path.join(output_folder, 'sac-save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # 创建训练环境和评估环境
    if not multiagent:
        # 单智能体环境
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=NUM_ENV,
                                 seed=0)
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        # 多智能体环境
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=NUM_ENV,
                                 seed=0)
        # 使用 Monitor 包裹，便于评估成功率
        eval_env = Monitor(MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT), info_keywords=("is_success",))

    # 打印环境空间信息
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # 定义 SAC 模型结构（自定义三层256单元 ReLU - SAC 通常需要更大的网络）
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])
    )
    
    # 创建 SAC 智能体
    # SAC 的关键超参数：
    # - buffer_size: Replay buffer 大小（off-policy 算法的关键）
    # - learning_starts: 开始学习前的随机探索步数
    # - batch_size: 每次更新的批量大小
    # - tau: 目标网络软更新系数
    # - gamma: 折扣因子
    # - train_freq: 训练频率
    model = SAC('MlpPolicy',
                train_env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                buffer_size=1000000,  # 1M replay buffer
                learning_starts=10000,  # 前10k步随机探索
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                tensorboard_log=filename+'/tb/',
                verbose=1, 
                device=CUDA)

    # 设定奖励阈值（达到则提前停止训练）
    target_reward = 949.5
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    # 评估回调，每5000步评估一次，保存最佳模型
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(5000),
                                 n_eval_episodes=10,
                                 deterministic=True,
                                 render=False)

    # 开始训练（本地1百万步，快速测试100步）
    # 注意：SAC 通常需要较多步数才能收敛，但样本效率比 PPO 高
    model.learn(total_timesteps=int(1e6) if local else int(1e2),
                callback=eval_callback,
                log_interval=10,
                progress_bar=True)

    # 保存最终模型
    model.save(filename+'/final_model.zip')
    print(f"[INFO] Model saved to: {filename}")

    # 输出训练过程评估结果
    if os.path.exists(filename+'/evaluations.npz'):
        with np.load(filename+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    # 本地训练时等待用户确认
    if local:
        input("Press Enter to continue to testing...")

    # 加载最佳模型（如无则报错）
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no best_model under the specified path", filename)
        print("[INFO]: using final model instead")
        path = filename+'/final_model.zip'
    
    model = SAC.load(path)

    # 创建测试环境（带GUI和不带GUI）
    if not multiagent:
        test_env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    
    # 日志记录器
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                   num_drones=DEFAULT_AGENTS if multiagent else 1,
                   output_folder=output_folder,
                   colab=colab)

    # 评估模型在测试环境的平均奖励
    print("\n[INFO] Evaluating trained SAC model...")
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print(f"\n[RESULT] Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")

    # 运行测试回合，记录并渲染
    print("[INFO] Running test episode with GUI...")
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        # 与 PPO 实现保持一致：去掉可能的中间维度（例如 vec env 带来的 batch 维度）
        # 这样 action 的形状在后续处理时更可预测
        try:
            action = np.squeeze(action, axis=1)
        except Exception:
            pass
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print(f"Step {i}: Obs shape={obs.shape}, Action={action}, Reward={reward:.3f}, Term={terminated}, Trunc={truncated}")
        
        # 记录日志（动力学观测时）
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                           timestamp=i/test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                           control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                               timestamp=i/test_env.CTRL_FREQ,
                               state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                               control=np.zeros(12))
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            print(f"[INFO] Episode terminated at step {i}")
            obs, info = test_env.reset(seed=42, options={})
    test_env.close()

    # 绘制训练/测试曲线
    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    # 命令行参数解析，支持多智能体、GUI、视频录制、保存路径等配置
    parser = argparse.ArgumentParser(description='单智能体/多智能体 SAC 强化学习示例脚本')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='是否使用多智能体环境（默认False）', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='是否显示PyBullet GUI（默认True）', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='是否录制视频（默认False）', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='日志/模型保存路径（默认results）', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='是否在Colab环境运行（默认False）', metavar='')
    parser.add_argument('--local',              default=True,                  type=str2bool,      help='是否本地长时间训练（默认True）', metavar='')
    ARGS = parser.parse_args()

    # 启动主流程
    run(**vars(ARGS))
