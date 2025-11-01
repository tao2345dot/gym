"""
无人机强化学习模型演示脚本

本脚本用于演示已训练好的PPO模型在无人机环境中的表现。
专门用于模型测试和可视化，不包含训练功能。

使用方法：
    $ conda activate drones
    $ python rolloutCP.py --multiagent false  # 单无人机演示
    $ python rolloutCP.py --multiagent true   # 多无人机演示

说明：
- 自动查找最新的训练结果文件夹
- 加载best_model.zip进行演示
- 支持单无人机和多无人机模式
- 包含详细的调试信息和中文注释
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
from datetime import datetime

# 强化学习相关导入
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# gym-pybullet-drones相关导入
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# 默认参数设置
DEFAULT_GUI = True  # 是否显示PyBullet GUI界面
DEFAULT_RECORD_VIDEO = False  # 是否录制演示视频
DEFAULT_OUTPUT_FOLDER = 'results'  # 模型结果文件夹路径
DEFAULT_COLAB = False  # 是否在Colab环境运行

DEFAULT_OBS = ObservationType('kin')  # 观测类型：'kin'（动力学）或 'rgb'（图像）
DEFAULT_ACT = ActionType('rpm')  # 动作类型：'rpm'（转速）/'pid'（PID控制）等
DEFAULT_AGENTS = 2  # 多无人机模式时的无人机数量
DEFAULT_MA = False  # 默认单无人机模式

# 无人机目标位置设置（可自定义）
DEFAULT_TARGET_POS = [0.8, 0, 1]  # 默认目标位置 [x, y, z]（米）
# 无人机起始位置设置（可自定义）
DEFAULT_INIT_POS = [0, 0, 0.1]  # 默认起始位置 [x, y, z]（米）
DEFAULT_MAX_STEPS = None  # 最大演示步数（None表示使用环境默认值）
DEFAULT_STOP_ON_TARGET = False  # 是否到达目标后停止演示

def get_latest_result_folder(output_folder):
    """
    自动查找最新的训练结果文件夹
    
    参数:
        output_folder: 结果文件夹根目录
        
    返回:
        str: 最新结果文件夹的完整路径
        
    异常:
        FileNotFoundError: 未找到任何结果文件夹时抛出
    """
    print(f"[调试] 正在查找结果文件夹: {output_folder}")
    
    # 查找所有save-*格式的文件夹
    folders = glob.glob(os.path.join(output_folder, 'save-*'))
    print(f"[调试] 找到的文件夹: {folders}")
    
    if not folders:
        raise FileNotFoundError(f"未找到任何结果文件夹于 {output_folder}")
    
    # 按修改时间排序，取最新的
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"[调试] 最新文件夹: {latest_folder}")
    
    return latest_folder

def load_model(result_folder):
    """
    加载训练好的PPO模型
    
    参数:
        result_folder: 结果文件夹路径
        
    返回:
        PPO: 加载的PPO模型对象
        
    异常:
        FileNotFoundError: 模型文件不存在时抛出
    """
    print(f"[调试] 正在从文件夹加载模型: {result_folder}")
    
    # 优先加载best_model.zip，其次是final_model.zip
    best_model_path = os.path.join(result_folder, 'best_model.zip')
    final_model_path = os.path.join(result_folder, 'final_model.zip')
    
    if os.path.isfile(best_model_path):
        model_path = best_model_path
        print(f"[调试] 使用最佳模型: {model_path}")
    elif os.path.isfile(final_model_path):
        model_path = final_model_path
        print(f"[调试] 使用最终模型: {model_path}")
    else:
        raise FileNotFoundError(f"[错误] 未找到模型文件于 {result_folder}")
    
    # 加载PPO模型
    try:
        model = PPO.load(model_path)
        print(f"[调试] 模型加载成功: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"[错误] 模型加载失败: {e}")

def create_test_environment(multiagent, gui, record_video, target_pos=DEFAULT_TARGET_POS, init_pos=DEFAULT_INIT_POS):
    """
    创建测试环境
    
    参数:
        multiagent: 是否多无人机模式
        gui: 是否显示GUI
        record_video: 是否录制视频
        target_pos: 目标位置 [x, y, z]
        init_pos: 起始位置 [x, y, z]
        
    返回:
        tuple: (测试环境, 无GUI测试环境)
    """
    print(f"[调试] 创建测试环境 - 多无人机模式: {multiagent}, GUI: {gui}")
    print(f"[调试] 目标位置设置为: {target_pos}")
    print(f"[调试] 起始位置设置为: {init_pos}")
    
    # 准备起始位置数组
    if not multiagent:
        # 单无人机：使用指定起始位置
        init_xyzs = np.array([init_pos])
        init_rpys = np.array([[0, 0, 0]])  # 初始姿态为水平
    else:
        # 多无人机：为每个无人机设置起始位置（在指定位置周围小范围分布）
        init_xyzs = np.array([
            [init_pos[0] + i * 0.5, init_pos[1] + i * 0.5, init_pos[2]] 
            for i in range(DEFAULT_AGENTS)
        ])
        init_rpys = np.array([[0, 0, 0] for _ in range(DEFAULT_AGENTS)])
    
    if not multiagent:
        # 单无人机环境
        print(f"[调试] 创建单无人机环境 (HoverAviary)")
        test_env = HoverAviary(
            gui=gui,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video,
            initial_xyzs=init_xyzs,
            initial_rpys=init_rpys
        )
        test_env_nogui = HoverAviary(
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            initial_xyzs=init_xyzs,
            initial_rpys=init_rpys
        )
        # 设置目标位置
        test_env.TARGET_POS = np.array(target_pos)
        test_env_nogui.TARGET_POS = np.array(target_pos)
    else:
        # 多无人机环境
        print(f"[调试] 创建多无人机环境 (MultiHoverAviary) - 无人机数量: {DEFAULT_AGENTS}")
        test_env = MultiHoverAviary(
            gui=gui,
            num_drones=DEFAULT_AGENTS,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video,
            initial_xyzs=init_xyzs,
            initial_rpys=init_rpys
        )
        test_env_nogui = MultiHoverAviary(
            num_drones=DEFAULT_AGENTS,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            initial_xyzs=init_xyzs,
            initial_rpys=init_rpys
        )
        # 设置目标位置
        test_env.TARGET_POS = np.array(target_pos)
        test_env_nogui.TARGET_POS = np.array(target_pos)
    
    print(f"[调试] 环境创建完成")
    print(f"[调试] 动作空间: {test_env.action_space}")
    print(f"[调试] 观测空间: {test_env.observation_space}")
    
    return test_env, test_env_nogui

def evaluate_model_performance(model, test_env_nogui, multiagent):
    """
    评估模型性能
    
    参数:
        model: PPO模型
        test_env_nogui: 无GUI测试环境
        multiagent: 是否多无人机模式
    """
    print(f"[调试] 开始评估模型性能...")
    
    try:
        mean_reward, std_reward = evaluate_policy(
            model, 
            test_env_nogui, 
            n_eval_episodes=5,  # 减少评估轮数以加快速度
            deterministic=True
        )
        print(f"[评估结果] 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    except Exception as e:
        print(f"[警告] 模型评估失败: {e}")
        print(f"[调试] 跳过性能评估，继续演示...")

def print_position_summary(start_positions, final_positions, target_pos, init_pos_setting, multiagent):
    """
    打印位置总结信息
    
    参数:
        start_positions: 实际起始位置
        final_positions: 最终位置  
        target_pos: 目标位置
        init_pos_setting: 设置的起始位置
        multiagent: 是否多无人机模式
    """
    print("\n" + "="*70)
    print("🚁 无人机位置总结报告")
    print("="*70)
    
    print(f"🎯 目标位置: X={target_pos[0]:.3f}m, Y={target_pos[1]:.3f}m, Z={target_pos[2]:.3f}m")
    print(f"🏠 设置起始位置: X={init_pos_setting[0]:.3f}m, Y={init_pos_setting[1]:.3f}m, Z={init_pos_setting[2]:.3f}m")
    
    if multiagent:
        # 多无人机模式
        print(f"📊 无人机数量: {len(start_positions)}")
        
        for i, (start_pos, final_pos) in enumerate(zip(start_positions, final_positions)):
            print(f"\n--- 无人机 #{i} ---")
            print(f"🟢 起始位置: X={start_pos[0]:.3f}m, Y={start_pos[1]:.3f}m, Z={start_pos[2]:.3f}m")
            print(f"🔴 最终位置: X={final_pos[0]:.3f}m, Y={final_pos[1]:.3f}m, Z={final_pos[2]:.3f}m")
            
            # 计算距离
            start_to_final = np.linalg.norm(final_pos - start_pos)
            final_to_target = np.linalg.norm(final_pos - target_pos)
            
            print(f"📏 移动距离: {start_to_final:.3f}m")
            print(f"📐 距目标距离: {final_to_target:.3f}m")
            
            # 判断是否到达目标
            if final_to_target < 0.5:  # 0.5米容差
                print(f"✅ 状态: 已接近目标！")
            elif final_to_target < 1.0:  # 1米容差
                print(f"🟡 状态: 接近目标")
            else:
                print(f"🔴 状态: 距离目标较远")
    else:
        # 单无人机模式
        print(f"\n--- 单无人机演示 ---")
        print(f"🟢 起始位置: X={start_positions[0]:.3f}m, Y={start_positions[1]:.3f}m, Z={start_positions[2]:.3f}m")
        print(f"🔴 最终位置: X={final_positions[0]:.3f}m, Y={final_positions[1]:.3f}m, Z={final_positions[2]:.3f}m")
        
        # 计算距离
        start_to_final = np.linalg.norm(final_positions - start_positions)
        final_to_target = np.linalg.norm(final_positions - target_pos)
        
        print(f"📏 移动距离: {start_to_final:.3f}m")
        print(f"📐 距目标距离: {final_to_target:.3f}m")
        
        # 判断是否到达目标
        if final_to_target < 0.5:  # 0.5米容差
            print(f"✅ 状态: 已接近目标！")
        elif final_to_target < 1.0:  # 1米容差
            print(f"🟡 状态: 接近目标")
        else:
            print(f"🔴 状态: 距离目标较远")
    
    print("="*70)
    print("📋 演示总结完成")
    print("="*70)

def run_demonstration(model, test_env, multiagent, output_folder, plot, colab, 
                     max_steps=None, stop_on_target=False, 
                     target_tolerance=0.1, init_pos_setting=DEFAULT_INIT_POS):
    """
    运行模型演示
    
    参数:
        model: PPO模型
        test_env: 测试环境
        multiagent: 是否多无人机模式
        output_folder: 输出文件夹
        plot: 是否绘制结果
        colab: 是否Colab环境
        max_steps: 最大演示步数（None使用默认值）
        stop_on_target: 是否到达目标后停止
        target_tolerance: 到达目标的距离容差（米）
        init_pos_setting: 设置的起始位置
    """
    print(f"[调试] 开始运行模型演示...")
    
    # 创建日志记录器
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=DEFAULT_AGENTS if multiagent else 1,
        output_folder=output_folder,
        colab=colab
    )
    
    # 重置环境，开始演示
    obs, info = test_env.reset(seed=42, options={})
    start_time = time.time()
    
    print(f"[调试] 初始观测形状: {obs.shape}")
    print(f"[调试] 目标位置: {test_env.TARGET_POS}")
    
    # 记录起始位置
    if multiagent:
        start_positions = []
        for drone_id in range(test_env.NUM_DRONES):
            start_pos = test_env._getDroneStateVector(drone_id)[0:3]
            start_positions.append(start_pos.copy())
    else:
        start_positions = test_env._getDroneStateVector(0)[0:3].copy()
    
    print(f"[调试] 开始演示循环...")
    
    # 演示循环 - 运行一个完整的回合
    episode_length = max_steps if max_steps else (test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ
    print(f"[调试] 演示步数: {episode_length}")
    print(f"[调试] 到达目标停止: {stop_on_target}, 距离容差: {target_tolerance}m")
    
    def check_target_reached():
        """检查是否到达目标位置"""
        if not stop_on_target:
            return False
        
        try:
            if multiagent:
                # 多无人机：检查所有无人机是否都到达目标
                for drone_id in range(test_env.NUM_DRONES):
                    pos = test_env._getDroneStateVector(drone_id)[0:3]
                    distance = np.linalg.norm(pos - test_env.TARGET_POS)
                    if distance > target_tolerance:
                        return False
                return True
            else:
                # 单无人机：检查是否到达目标
                pos = test_env._getDroneStateVector(0)[0:3]
                distance = np.linalg.norm(pos - test_env.TARGET_POS)
                return distance <= target_tolerance
        except:
            return False
    
    for i in range(episode_length):
        try:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            
            # 调试：打印动作信息
            if i % 100 == 0:  # 每100步打印一次，避免输出过多
                print(f"[调试] 步数 {i}: 原始动作形状 = {action.shape}")
            
            # 动作形状处理
            if multiagent:
                # 多无人机：参考single_learn.py的处理方式
                if action.ndim > 2:  # 如果有多余维度则压缩
                    action = np.squeeze(action, axis=1)
                print(f"[调试] 多无人机动作处理后形状: {action.shape}") if i % 100 == 0 else None
            else:
                # 单无人机：确保动作是正确形状
                if action.ndim == 1:
                    action = action.reshape(1, -1)
                elif action.ndim > 2:
                    action = np.squeeze(action)
                    if action.ndim == 1:
                        action = action.reshape(1, -1)
                print(f"[调试] 单无人机动作处理后形状: {action.shape}") if i % 100 == 0 else None
            
            # 执行动作
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            # 调试信息（每100步打印一次）
            if i % 100 == 0:
                print(f"[调试] 步数 {i}: 奖励={reward:.3f}, 终止={terminated}, 截断={truncated}")
                print(f"[调试] 新观测形状: {obs.shape}")
                
                # 显示无人机当前位置与目标距离
                try:
                    if multiagent:
                        for drone_id in range(test_env.NUM_DRONES):
                            pos = test_env._getDroneStateVector(drone_id)[0:3]
                            distance = np.linalg.norm(pos - test_env.TARGET_POS)
                            print(f"[调试] 无人机{drone_id}: 位置={pos}, 距离目标={distance:.3f}m")
                    else:
                        pos = test_env._getDroneStateVector(0)[0:3]
                        distance = np.linalg.norm(pos - test_env.TARGET_POS)
                        print(f"[调试] 无人机位置: {pos}, 距离目标: {distance:.3f}m")
                except:
                    pass
            
            # 检查是否到达目标位置
            if check_target_reached():
                print(f"[信息] 🎯 无人机已到达目标位置！停止演示")
                break
            
            # 记录日志
            try:
                obs2 = obs.squeeze() if hasattr(obs, 'squeeze') else obs
                act2 = action.squeeze() if hasattr(action, 'squeeze') else action
                
                if DEFAULT_OBS == ObservationType.KIN:
                    if not multiagent:
                        # 单无人机日志记录
                        logger.log(
                            drone=0,
                            timestamp=i/test_env.CTRL_FREQ,
                            state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                            control=np.zeros(12)
                        )
                    else:
                        # 多无人机日志记录
                        for d in range(DEFAULT_AGENTS):
                            logger.log(
                                drone=d,
                                timestamp=i/test_env.CTRL_FREQ,
                                state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                                control=np.zeros(12)
                            )
            except Exception as log_error:
                if i % 100 == 0:
                    print(f"[警告] 日志记录失败: {log_error}")
            
            # 渲染环境
            test_env.render()
            
            # 时间同步
            sync(i, start_time, test_env.CTRL_TIMESTEP)
            
            # 如果回合结束则重置
            if terminated:
                print(f"[调试] 回合在步数 {i} 结束，重置环境")
                obs, info = test_env.reset(seed=42, options={})
                
        except Exception as step_error:
            print(f"[错误] 第 {i} 步执行失败: {step_error}")
            print(f"[调试] 观测形状: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"[调试] 动作形状: {action.shape if hasattr(action, 'shape') else type(action)}")
            break
    
    # 记录最终位置
    if multiagent:
        final_positions = []
        for drone_id in range(test_env.NUM_DRONES):
            final_pos = test_env._getDroneStateVector(drone_id)[0:3]
            final_positions.append(final_pos.copy())
    else:
        final_positions = test_env._getDroneStateVector(0)[0:3].copy()
    
    # 关闭环境
    test_env.close()
    print(f"[调试] 演示完成，环境已关闭")
    
    # 输出位置总结
    print_position_summary(start_positions, final_positions, test_env.TARGET_POS, init_pos_setting, multiagent)
    
    # 绘制结果（如果需要）
    if plot and DEFAULT_OBS == ObservationType.KIN:
        try:
            logger.plot()
            print(f"[调试] 结果绘制完成")
        except Exception as plot_error:
            print(f"[警告] 结果绘制失败: {plot_error}")

def main(multiagent=DEFAULT_MA, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, 
         output_folder=DEFAULT_OUTPUT_FOLDER, colab=DEFAULT_COLAB, plot=True,
         target_pos=DEFAULT_TARGET_POS, init_pos=DEFAULT_INIT_POS, max_steps=DEFAULT_MAX_STEPS, 
         stop_on_target=DEFAULT_STOP_ON_TARGET):
    """
    主函数：模型演示流程
    
    参数:
        multiagent: 是否多无人机模式
        gui: 是否显示GUI
        record_video: 是否录制视频
        output_folder: 输出文件夹
        colab: 是否Colab环境
        plot: 是否绘制结果
        target_pos: 目标位置 [x, y, z]
        init_pos: 起始位置 [x, y, z]
        max_steps: 最大演示步数
        stop_on_target: 是否到达目标后停止
    """
    print("="*60)
    print("无人机强化学习模型演示开始")
    print("="*60)
    
    try:
        # 步骤1：查找最新结果文件夹
        print("\n[步骤1] 查找最新训练结果...")
        result_folder = get_latest_result_folder(output_folder)
        
        # 步骤2：加载模型
        print("\n[步骤2] 加载训练好的模型...")
        model = load_model(result_folder)
        
        # 步骤3：创建测试环境
        print("\n[步骤3] 创建测试环境...")
        test_env, test_env_nogui = create_test_environment(multiagent, gui, record_video, target_pos, init_pos)
        
        # 步骤4：评估模型性能
        print("\n[步骤4] 评估模型性能...")
        evaluate_model_performance(model, test_env_nogui, multiagent)
        
        # 步骤5：运行演示
        print("\n[步骤5] 运行模型演示...")
        print("按 Ctrl+C 可随时停止演示")
        
        run_demonstration(model, test_env, multiagent, output_folder, plot, colab, 
                         max_steps, stop_on_target, init_pos_setting=init_pos)
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n[信息] 用户中断演示")
    except Exception as e:
        print(f"\n[错误] 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='无人机强化学习模型演示脚本')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool, 
                        help='是否使用多无人机模式（默认: False）', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, 
                        help='是否显示PyBullet GUI（默认: True）', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, 
                        help='是否录制演示视频（默认: False）', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='模型结果文件夹路径（默认: "results"）', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, 
                        help='是否在Colab环境运行（默认: False）', metavar='')
    parser.add_argument('--target_x', default=DEFAULT_TARGET_POS[0], type=float, 
                        help='目标位置X坐标（米，默认: 0）', metavar='')
    parser.add_argument('--target_y', default=DEFAULT_TARGET_POS[1], type=float, 
                        help='目标位置Y坐标（米，默认: 1）', metavar='')
    parser.add_argument('--target_z', default=DEFAULT_TARGET_POS[2], type=float, 
                        help='目标位置Z坐标（米，默认: 0）', metavar='')
    parser.add_argument('--init_x', default=DEFAULT_INIT_POS[0], type=float, 
                        help='起始位置X坐标（米，默认: 0）', metavar='')
    parser.add_argument('--init_y', default=DEFAULT_INIT_POS[1], type=float, 
                        help='起始位置Y坐标（米，默认: 0）', metavar='')
    parser.add_argument('--init_z', default=DEFAULT_INIT_POS[2], type=float, 
                        help='起始位置Z坐标（米，默认: 1）', metavar='')
    parser.add_argument('--max_steps', default=DEFAULT_MAX_STEPS, type=int, 
                        help='最大演示步数（默认: 环境默认值）', metavar='')
    parser.add_argument('--stop_on_target', default=DEFAULT_STOP_ON_TARGET, type=str2bool, 
                        help='是否到达目标后停止演示（默认: False）', metavar='')
    
    # 解析参数
    args = parser.parse_args()
    
    # 组合目标位置和起始位置
    target_pos = [args.target_x, args.target_y, args.target_z]
    init_pos = [args.init_x, args.init_y, args.init_z]
    
    # 打印运行配置
    print("运行配置:")
    print(f"  多无人机模式: {args.multiagent}")
    print(f"  显示GUI: {args.gui}")
    print(f"  录制视频: {args.record_video}")
    print(f"  结果文件夹: {args.output_folder}")
    print(f"  Colab环境: {args.colab}")
    print(f"  起始位置: {init_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  最大步数: {args.max_steps}")
    print(f"  到达目标停止: {args.stop_on_target}")
    
    # 启动演示
    main(
        multiagent=args.multiagent,
        gui=args.gui,
        record_video=args.record_video,
        output_folder=args.output_folder,
        colab=args.colab,
        target_pos=target_pos,
        init_pos=init_pos,
        max_steps=args.max_steps,
        stop_on_target=args.stop_on_target
    )
