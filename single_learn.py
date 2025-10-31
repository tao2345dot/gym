"""
本脚本演示了如何使用 `gym_pybullet_drones` 的 Gymnasium 接口进行强化学习。

主要功能：
1. 支持单智能体和多智能体环境（HoverAviary / MultiHoverAviary），用于 PPO 算法训练。
2. 集成 stable-baselines3 强化学习库，实现训练、评估、保存和测试。

使用方法：
    $ python single_learn.py --multiagent false  # 单智能体
    $ python single_learn.py --multiagent true   # 多智能体

说明：
这是一个最简可运行的 gym-pybullet-drones 与 stable-baselines3 集成示例。
"""
import sys
import os
# 添加父目录到 sys.path，便于导入 gym_pybullet_drones 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
# 导入 stable-baselines3 相关模块
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor  # custom ---v
from PIL import Image  # custom ---^

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# custom ---v
# midify before every experiment
# NUM_ENV = 64  # Original setting for high-end GPUs
NUM_ENV = 32  # Recommended for RTX 4050 (6GB VRAM)
EXP_NAME = "objin128bs1r0p3cb_diff128x128x128_30cm_1drone_rtx4050"
CUDA = "cuda:0"
# custom ---^

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
    主流程函数：训练、评估、保存、测试强化学习模型。
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
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
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

    # 定义 PPO 模型结构（自定义三层128单元ReLU）
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]))
    # 创建 PPO 智能体
    model = PPO('MlpPolicy',
                train_env,
                policy_kwargs=policy_kwargs,
                learning_rate=0.0003,
                batch_size=64,   # RTX 4050 显存优化
                tensorboard_log=filename+'/tb/',
                verbose=1, 
                device=CUDA)

    # 设定奖励阈值（达到则提前停止训练）
    target_reward = 949.5
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    # 评估回调，每1000步评估一次，保存最佳模型
    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 n_eval_episodes=10,
                                 deterministic=True,
                                 render=False)

    # 开始训练（本地3百万步，快速测试100步）
    model.learn(total_timesteps=int(3e6) if local else int(1e2),
                callback=eval_callback,
                log_interval=100,
                progress_bar=True)

    # 保存最终模型
    model.save(filename+'/final_model.zip')
    print(filename)

    # 输出训练过程评估结果
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    # 本地训练时等待用户确认
    if local:
        input("Press Enter to continue...")

    # 加载最佳模型（如无则报错）
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

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
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    # 运行测试回合，记录并渲染
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        # 多智能体时 squeeze 处理动作
        action = np.squeeze(action, axis=1)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
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
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    # 绘制训练/测试曲线
    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    # 命令行参数解析，支持多智能体、GUI、视频录制、保存路径等配置
    parser = argparse.ArgumentParser(description='单智能体/多智能体强化学习示例脚本')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='是否使用多智能体环境（默认False）', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='是否显示PyBullet GUI（默认True）', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='是否录制视频（默认False）', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='日志/模型保存路径（默认results）', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='是否在Colab环境运行（默认False）', metavar='')
    ARGS = parser.parse_args()

    # 启动主流程
    run(**vars(ARGS))
