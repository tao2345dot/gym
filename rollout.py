"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
# custom ---v
from stable_baselines3.common.monitor import Monitor
from PIL import Image
# custom ---^

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# custom ---v
# midify before every experiment
# NUM_ENV = 64
# EXP_NAME = "objin128bs1r0p3cb_diff128x128x128_30cm_1drone"
# CUDA = "cuda:0"
NUM_ENV = 32  # Recommended for RTX 4050 (6GB VRAM)
EXP_NAME = "objin128bs1r0p3cb_diff128x128x128_30cm_1drone_rtx4050"
CUDA = "cuda:0"
# custom ---^

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 8
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    # if not os.path.exists(filename):
    #     os.makedirs(filename+'/')

    # if not multiagent:
    #     train_env = make_vec_env(HoverAviary,
    #                              env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
    #                              n_envs=NUM_ENV,
    #                              seed=0
    #                              )
    #     eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    # else:
    #     train_env = make_vec_env(MultiHoverAviary,
    #                              env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
    #                              n_envs=NUM_ENV,
    #                              seed=0
    #                              )
    #     # eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    #     # custom ---v
    #     eval_env = Monitor(MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT), info_keywords=("is_success",))
    #     # custom ---^

    # #### Check the environment's spaces ########################
    # print('[INFO] Action space:', train_env.action_space)
    # print('[INFO] Observation space:', train_env.observation_space)

    # #### Train the model #######################################
    # # model = PPO('MlpPolicy',
    # #             train_env,
    # #             # tensorboard_log=filename+'/tb/',
    # #             verbose=1)

    # # custom ---v
    # # Custom actor (pi) and value function (vf) networks
    # # of two layers of size 32 each with Relu activation function
    # # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                     net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]))
    # # Create the agent
    # # model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
    
    # model = PPO('MlpPolicy',
    #             train_env,
    #             policy_kwargs=policy_kwargs,
    #             learning_rate=0.0003,
    #             batch_size=128,
    #             tensorboard_log=filename+'/tb/',
    #             verbose=1, 
    #             device=CUDA)
    # # custom ---^  

    # #### Target cumulative rewards (problem-dependent) ##########
    # # if DEFAULT_ACT == ActionType.ONE_D_RPM:
    # #     target_reward = 474.15 if not multiagent else 949.5
    # # else:
    # #     target_reward = 467. if not multiagent else 920.
    # # custom ---v
    # target_reward = 949.5
    # # custom ---^
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
    #                                                  verbose=1)
    # eval_callback = EvalCallback(eval_env,
    #                             #  callback_on_new_best=callback_on_best,
    #                              verbose=1,
    #                              best_model_save_path=filename+'/',
    #                              log_path=filename+'/',
    #                              eval_freq=int(1000),
    #                              n_eval_episodes=10,  # Number of episodes to evaluate
    #                              deterministic=True,
    #                              render=False)
    # model.learn(total_timesteps=int(1e7) if local else int(1e2), # shorter training in GitHub Actions pytest
    #             callback=eval_callback,
    #             log_interval=100,
    #             progress_bar = True)

    # #### Save the model ########################################
    # model.save(filename+'/final_model.zip')
    # print(filename)

    #### Print training progression ############################
    # 原始代码：硬编码评估结果路径，容易导致找不到文件
    # filename = os.path.join(output_folder,'save-07.03.2025_15.18.53')
    # with np.load(filename+'/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    # 新代码：自动查找最新的结果文件夹，并读取评估结果
    import glob
    def get_latest_result_folder(output_folder):
        """
        自动查找 output_folder 下最新的 save-* 结果文件夹。
        返回最新文件夹路径。
        """
        folders = glob.glob(os.path.join(output_folder, 'save-*'))
        if not folders:
            raise FileNotFoundError(f"未找到任何结果文件夹于 {output_folder}")
        # 按修改时间排序，取最新
        return max(folders, key=os.path.getmtime)

    filename = get_latest_result_folder(output_folder)
    # 读取最新结果文件夹下的评估结果
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    # 参考 single_learn.py：直接加载一个模型，所有无人机共享同一个策略
    model = PPO.load(path)
    # 为了兼容原有代码，也保存到 model_list 中
    model_list = [model]

    print("TEST: model loaded.==================================")
    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    # mean_reward, std_reward = evaluate_policy(model,
    #                                           test_env_nogui,
    #                                           n_eval_episodes=10
    #                                           )
    # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    # 参考 single_learn.py 的测试逻辑，统一使用一个模型处理
    if not multiagent:
        # 单智能体情况：使用第一个模型
        model = model_list[0]
    else:
        # 多智能体情况：也使用第一个模型（所有无人机共享同一个策略）
        model = model_list[0]
    
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        
        # 确保动作形状正确：单智能体时需要 (1, 4)，多智能体时需要 (n, 4)
        if not multiagent:
            # 单智能体：确保动作是 (1, 4) 形状
            if action.ndim == 1:
                action = action.reshape(1, -1)
        else:
            # 多智能体：参考 single_learn.py 的处理方式
            action = np.squeeze(action, axis=1)
        
        print(f"DEBUG: action shape = {action.shape}")  # 调试信息
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        
        # 记录日志（参考 single_learn.py）
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
            obs, info = test_env.reset(seed=42, options={})
    
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
