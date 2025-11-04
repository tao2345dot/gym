"""
NavigationAviary - 支持随机目标导航的强化学习环境

关键特性：
1. 观测空间包含目标位置（15维 = 12维状态 + 3维目标）
2. 每个 episode 随机采样目标点
3. 奖励函数鼓励到达目标并保持稳定
"""
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.circleTask_BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class NavigationAviary(BaseRLAviary):
    """支持随机目标导航的单智能体 RL 环境"""
    
    # 类变量：设置环境规格（禁用 TimeLimit wrapper）
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """初始化导航环境"""
        # 如果没有指定初始位置，使用更高的默认高度（1.0m）
        if initial_xyzs is None:
            initial_xyzs = np.array([[0, 0, 1.0]])
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,  # 单智能体
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )
        
        # 禁用 Gymnasium 的 TimeLimit wrapper
        # 设置 spec 为 None 或设置一个大的 max_episode_steps
        self.spec = None
        
        # 目标位置将在 reset() 中随机生成
        self.TARGET_POS = None
        
        # 导航任务的空间范围（比训练时略小，确保安全）
        self.TASK_SPACE_LOW = np.array([-1.2, -1.2, 0.3])
        self.TASK_SPACE_HIGH = np.array([1.2, 1.2, 2.0])
        
        # 成功判定阈值（根据实际测试调整）
        # 测试显示模型能稳定到 13-22cm，0.1m 过于严格
        self.SUCCESS_THRESHOLD = 0.15  # 距离目标 < 15cm 算成功
        self.SUCCESS_HOLD_STEPS = 30  # 需要保持 1 秒（30步 @ 30Hz）
        self.success_counter = 0

    def reset(self, seed=None, options=None):
        """重置环境并随机生成新目标"""
        # 先生成目标位置（在调用 super().reset() 之前，因为它会调用 _computeObs()）
        if seed is not None:
            np.random.seed(seed)
        
        # 随机生成目标位置（避开起点附近）
        while True:
            self.TARGET_POS = np.random.uniform(
                low=self.TASK_SPACE_LOW,
                high=self.TASK_SPACE_HIGH,
                size=(1, 3)
            )
            # 确保目标距离起点至少 0.5m（避免太简单）
            # 如果 INIT_XYZS 还没初始化，使用默认起点 [0, 0, 1]
            init_pos = self.INIT_XYZS if hasattr(self, 'INIT_XYZS') else np.array([[0, 0, 1]])
            dist = np.linalg.norm(self.TARGET_POS - init_pos)
            if dist >= 0.5:
                break
        
        self.success_counter = 0
        
        # 然后调用父类 reset（会调用 _computeObs，此时 TARGET_POS 已就绪）
        obs, info = super().reset(seed=seed, options=options)
        
        print(f"[NavigationAviary] New target: {self.TARGET_POS[0]}, dist: {dist:.2f}m")
        
        return obs, info

    def _observationSpace(self):
        """
        定义观测空间：15维
        - [0:3]   位置 (x, y, z)
        - [3:6]   姿态 (roll, pitch, yaw)
        - [6:9]   速度 (vx, vy, vz)
        - [9:12]  角速度 (wx, wy, wz)
        - [12:15] 目标位置 (target_x, target_y, target_z)
        """
        # 状态空间范围（与父类一致）
        lo = -np.inf
        hi = np.inf
        
        # 前12维：状态（位置/姿态/速度/角速度）
        obs_lower_bound = np.array([
            lo, lo, 0,           # 位置
            lo, lo, lo,          # 姿态
            lo, lo, lo,          # 速度
            lo, lo, lo,          # 角速度
            lo, lo, 0            # 目标位置（z >= 0）
        ])
        
        obs_upper_bound = np.array([
            hi, hi, hi,          # 位置
            hi, hi, hi,          # 姿态
            hi, hi, hi,          # 速度
            hi, hi, hi,          # 角速度
            hi, hi, hi           # 目标位置
        ])
        
        return spaces.Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            dtype=np.float32
        )

    def _computeObs(self):
        """
        计算观测：12维状态 + 3维目标位置
        
        Returns:
            ndarray: shape (1, 15) 的观测数组
        """
        # 获取无人机状态（20维）
        state = self._getDroneStateVector(0)
        
        # 提取关键状态（12维）
        # state[0:3]   = 位置
        # state[7:10]  = 姿态（roll, pitch, yaw）
        # state[10:13] = 速度
        # state[13:16] = 角速度
        obs_12 = np.hstack([
            state[0:3],    # 位置
            state[7:10],   # 姿态
            state[10:13],  # 速度
            state[13:16]   # 角速度
        ])
        
        # 拼接目标位置（3维）
        obs_15 = np.hstack([obs_12, self.TARGET_POS.flatten()])
        
        return obs_15.reshape(1, 15).astype('float32')

    def _computeReward(self):
        """
        计算奖励：鼓励到达目标并保持稳定
        
        奖励组成：
        1. 距离惩罚：未到达时每步 -0.1
        2. 接近奖励：靠近目标时给予 shaping reward
        3. 成功奖励：到达并保持时给予大奖励
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        
        dist = np.linalg.norm(self.TARGET_POS - pos)
        
        # 基础惩罚：鼓励快速完成任务
        reward = -0.1
        
        # 分层距离 shaping（逐步细化）
        if dist < 0.5:
            # 距离 0.5m 内：奖励 0~1.0
            reward += (0.5 - dist) * 2.0
        
        if dist < 0.2:
            # 距离 0.2m 内：额外奖励 0~2.0
            reward += (0.2 - dist) * 10.0
        
        if dist < 0.15:
            # 距离 0.15m 内：超高奖励 0~3.75
            # 解决模型卡在 13-22cm 的问题
            reward += (0.15 - dist) * 25.0
        
        # 成功到达并保持
        if dist < self.SUCCESS_THRESHOLD:
            self.success_counter += 1
            reward += 2.0  # 在目标区域内持续奖励（增强）
            if self.success_counter >= self.SUCCESS_HOLD_STEPS:
                reward += 100.0  # 成功完成任务：大额奖励
        else:
            self.success_counter = 0
        
        # 姿态惩罚：倾斜过大时惩罚
        tilt = np.sqrt(rpy[0]**2 + rpy[1]**2)
        if tilt > 0.5:  # 超过 ~30 度
            reward -= 1.0
        
        return reward

    def _computeTerminated(self):
        """
        判定是否因成功/失败而终止
        
        终止条件：
        1. 成功到达并保持
        2. 飞出边界
        3. 坠毁（高度过低）
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # 成功到达
        if self.success_counter >= self.SUCCESS_HOLD_STEPS:
            return True
        
        # 飞出边界（比训练空间略大）
        if (pos[0] < -1.5 or pos[0] > 1.5 or
            pos[1] < -1.5 or pos[1] > 1.5):
            return True
        
        if pos[2] < 0.05 or pos[2] > 2.5:
            return True
        
        return False

    def _computeTruncated(self):
        """
        判定是否因超时而截断
        
        超时条件：
        - 超过最大步数（默认 8 秒）
        
        注意：step_counter 是物理步数（240Hz），不是控制步数（30Hz）
        8秒 = 8 * 240 = 1920 物理步
        """
        if self.step_counter > 8 * self.PYB_FREQ:  # 8 秒超时
            return True
        
        # 姿态失控（倾斜过大）
        state = self._getDroneStateVector(0)
        rpy = state[7:10]
        if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:  # > 46度
            return True
        
        return False

    def _computeInfo(self):
        """
        返回额外信息（用于调试和评估）
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        
        return {
            "distance_to_target": dist,
            "is_success": self.success_counter >= self.SUCCESS_HOLD_STEPS,
            "target_pos": self.TARGET_POS[0].copy()
        }
