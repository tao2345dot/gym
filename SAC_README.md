# SAC 算法无人机飞行 + LLM 避障

本目录包含基于 SAC (Soft Actor-Critic) 算法的无人机飞行训练和 LLM 避障集成示例。

## 文件说明

### 1. `sac_learn.py`
SAC 算法训练脚本，用于训练无人机悬停/导航任务。

**主要特点：**
- 使用 SAC 算法（off-policy，样本高效）
- 支持单智能体和多智能体环境
- 自定义神经网络架构（3层256单元）
- Replay buffer 大小：1M
- 自动保存最佳模型

**使用方法：**

```bash
# 激活虚拟环境
source myenv/bin/activate

# 单智能体训练（默认）
python -m gym_pybullet_drones.custom.sac_learn --local true --gui false

# 多智能体训练
python -m gym_pybullet_drones.custom.sac_learn --multiagent true --local true --gui false

# 快速测试（100步）
python -m gym_pybullet_drones.custom.sac_learn --local false --gui true
```

**训练参数：**
- 训练步数：1,000,000 步（本地训练）
- 评估频率：每 5,000 步
- Replay buffer：1M
- Batch size：256
- 学习率：3e-4
- 并行环境数：16

**输出：**
- 模型文件：`results/sac-save-<timestamp>/best_model.zip`
- TensorBoard 日志：`results/sac-save-<timestamp>/tb/`
- 评估数据：`results/sac-save-<timestamp>/evaluations.npz`

### 2. `sac_llm_obstacle_avoidance.py`
SAC + LLM 避障集成脚本，演示如何将训练好的 SAC 模型与 LLM 避障规划器结合。

**主要特点：**
- 加载预训练的 SAC 模型
- 实时检测障碍物碰撞
- 使用 LLM（OpenAI）或几何规划生成避障路径点
- 可视化障碍物和路径点
- 状态机管理（to_target / avoiding）

**使用方法：**

```bash
# 首先训练 SAC 模型（或使用已有模型）
python -m gym_pybullet_drones.custom.sac_learn --local true --gui false

# 运行避障演示（使用 LLM）
export OPENAI_API_KEY='your-api-key-here'
python -m gym_pybullet_drones.custom.sac_llm_obstacle_avoidance \
    --model_path results/sac-save-01.31.2025_12.00.00/best_model.zip \
    --gui true \
    --use_llm true

# 运行避障演示（仅几何规划，不需要 API Key）
python -m gym_pybullet_drones.custom.sac_llm_obstacle_avoidance \
    --model_path results/sac-save-01.31.2025_12.00.00/best_model.zip \
    --gui true \
    --use_llm false
```

**参数说明：**
- `--model_path`: SAC 模型文件路径（必需）
- `--gui`: 是否显示 PyBullet GUI（默认 True）
- `--use_llm`: 是否使用 LLM 规划（默认 True）
- `--openai_model`: OpenAI 模型名称（默认 gpt-3.5-turbo）
- `--record_video`: 是否录制视频（默认 False）

## SAC vs PPO 对比

| 特性 | SAC | PPO |
|------|-----|-----|
| 算法类型 | Off-policy | On-policy |
| 样本效率 | 高（使用 replay buffer） | 低（需要更多交互数据） |
| 训练稳定性 | 较稳定 | 非常稳定 |
| 计算开销 | 中等 | 低 |
| 并行环境数 | 较少（16） | 较多（32-64） |
| 收敛速度 | 较快 | 较慢 |
| 适用场景 | 连续动作空间 | 连续/离散动作空间 |

## 避障逻辑说明

### 状态机设计

```
初始状态: to_target
    |
    v
检测到障碍物?
    |
    ├─ 是 ──> 状态: avoiding
    |          ├─ 调用 plan_avoidance() 生成避障路径点
    |          ├─ 添加路径点可视化标记
    |          └─ SAC 模型执行导航
    |               |
    |               v
    |          到达路径点?
    |               |
    |               └─ 是 ──> 状态: to_target
    |
    └─ 否 ──> 继续朝目标前进
```

### LLM 避障规划

当检测到障碍物时，系统会调用 `plan_avoidance()` 函数：

1. **优先使用 LLM**（如果 `OPENAI_API_KEY` 可用）：
   - 构建包含环境边界、障碍物信息、约束条件的提示词
   - 请求 GPT 生成安全的避障路径点
   - 解析 JSON 响应获取 3D 坐标

2. **回退到几何规划**（如果 LLM 不可用或失败）：
   - 使用垂直于障碍物-智能体连线的方向
   - 选择距离目标更近的候选点
   - 保持与障碍物的安全距离

### 碰撞检测

- 在 XY 平面检测与障碍物的距离
- 考虑障碍物半径 + 安全余量（默认 0.2m）
- 检查 Z 轴高度是否在障碍物范围内

## 环境配置

### 障碍物设置

在 `sac_llm_obstacle_avoidance.py` 中修改 `OBSTACLES` 列表：

```python
OBSTACLES = [
    {'pos': [0.5, 0.5, 0.5], 'radius': 0.3, 'height': 1.0},
    {'pos': [-0.5, 0.5, 0.5], 'radius': 0.25, 'height': 0.8},
    # 添加更多障碍物...
]
```

### 目标位置

修改 `TARGET_POS` 变量：

```python
TARGET_POS = np.array([1.0, 1.0, 1.0])  # [x, y, z]
```

## 依赖项

确保已安装以下库（已在 `myenv` 中安装）：

```bash
pip install numpy scipy matplotlib pybullet gymnasium stable-baselines3 torch tensorboard
```

如果使用 LLM 避障，还需要：

```bash
pip install openai
```

## TensorBoard 可视化

查看训练过程：

```bash
tensorboard --logdir results/sac-save-<timestamp>/tb/
```

在浏览器中打开 `http://localhost:6006`

## 故障排除

### 1. ImportError: tensorboard not installed
```bash
./myenv/bin/python -m pip install tensorboard
```

### 2. CUDA out of memory
减少 `NUM_ENV` 或 `batch_size`：
```python
NUM_ENV = 8  # 减少并行环境数
batch_size=128  # 减少批量大小
```

### 3. 模型不收敛
- 增加训练步数：`total_timesteps=int(2e6)`
- 调整学习率：`learning_rate=1e-4`
- 增加 replay buffer：`buffer_size=2000000`

### 4. LLM 规划失败
检查：
- `OPENAI_API_KEY` 是否设置正确
- 网络连接是否正常
- API 配额是否充足

系统会自动回退到几何规划，不会中断运行。

## 进阶用法

### 自定义奖励函数

修改环境文件 `gym_pybullet_drones/envs/obsin_HoverAviary.py` 中的 `_computeReward()` 方法。

### 修改 SAC 超参数

在 `sac_learn.py` 中修改 SAC 初始化参数：

```python
model = SAC('MlpPolicy',
            train_env,
            learning_rate=1e-4,  # 学习率
            buffer_size=2000000,  # Replay buffer 大小
            learning_starts=20000,  # 开始学习前的随机探索步数
            batch_size=512,  # 批量大小
            tau=0.01,  # 目标网络更新系数
            gamma=0.995,  # 折扣因子
            ent_coef='auto',  # 熵系数（自动调整）
            ...)
```

### 多目标导航

修改 `sac_llm_obstacle_avoidance.py`，添加多个目标点列表并循环导航。

## 参考资料

- [Stable-Baselines3 SAC 文档](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SAC 论文](https://arxiv.org/abs/1801.01290)
- [gym-pybullet-drones 文档](https://github.com/utiasDSL/gym-pybullet-drones)

### 3. `sac_continuous_navigator.py`
SAC 多障碍物环境连续导航系统，支持多目标点顺序导航和智能避障。

**主要特点：**
- 基于 SAC 算法的连续导航控制
- 支持多障碍物环境
- 实时碰撞检测和避障规划
- LLM 或几何避障两种模式
- 可视化障碍物、目标和避障路径点
- 键盘和网络双重控制接口
- 自动轨迹记录和可视化

**使用方法：**

```bash
# 首先训练 SAC 模型
python -m gym_pybullet_drones.custom.sac_learn --local true --gui false

# 启动连续导航（使用 LLM 避障）
export OPENAI_API_KEY='your-api-key-here'
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm true

# 启动连续导航（仅几何避障）
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm false
```

**控制方式：**

1. **键盘控制**
   - `空格键`: 暂停/继续
   - `H 键`: 返回起点
   - `Q/ESC`: 退出程序

2. **网络控制** (TCP 端口 8888)
   ```python
   import socket
   import json
   
   # 发送目标点
   command = {'type': 'target', 'target': [1.0, 1.0, 1.0]}
   s = socket.socket()
   s.connect(('localhost', 8888))
   s.send(json.dumps(command).encode())
   response = s.recv(1024)
   s.close()
   ```

**障碍物配置：**

系统使用 4 根固定大小、随机位置的柱子作为障碍物。在 `sac_continuous_navigator.py` 中修改 `PILLAR_CONFIG`：

```python
# 柱子固定配置
PILLAR_CONFIG = {
    'radius': 0.08,      # 柱子半径（米）
    'height': 2.0,       # 柱子高度（米）
    'count': 4,          # 柱子数量
    'color': [0.6, 0.6, 0.6, 1.0],  # 灰色柱子
}
```

柱子位置在每次启动时随机生成，满足以下约束：
- X, Y 坐标范围：[-1.5, 1.5]
- 柱子之间最小距离：0.8m
- 原点安全半径：0.5m（避免柱子生成在起点）

**避障参数：**

```python
COLLISION_THRESHOLD = 0.5  # 碰撞检测距离阈值（柱子半径 + 安全余量）
WAYPOINT_REACH_THRESHOLD = 0.15  # 路径点到达阈值
TARGET_REACH_THRESHOLD = 0.2  # 目标点到达阈值
```

**随机柱子生成参数：**

```python
# 在 generate_random_pillars 函数中
x_range=(-1.5, 1.5)           # X 坐标范围
y_range=(-1.5, 1.5)           # Y 坐标范围
min_distance=0.8              # 柱子之间最小距离
origin_safe_radius=0.5        # 原点安全半径
```

**输出：**
- 实时状态显示（位置、目标、距离、模式）
- 轨迹可视化图（3D轨迹、XY平面投影、高度变化）
- 统计信息（运行时长、到达目标数、避障次数）

## SAC 连续导航工作流程

```
初始化系统
    ↓
加载 SAC 模型
    ↓
创建扩展环境（含障碍物）
    ↓
主循环开始 ────────────────┐
    ↓                      │
检测碰撞风险？             │
    ├─ 是 → 规划避障路径点  │
    │       ↓              │
    │   导航到路径点        │
    │       ↓              │
    │   到达路径点？        │
    │       ├─ 是 → 恢复   │
    │       └─ 否 ─────────┤
    │                      │
    └─ 否 → 直接导航到目标  │
            ↓              │
        到达目标？          │
            ├─ 是 → 下一目标
            └─ 否 ─────────┘
```

## SAC 连续导航 vs PPO 连续导航

| 特性 | SAC 连续导航 | PPO 连续导航 |
|------|-------------|-------------|
| 算法 | SAC (Off-policy) | PPO (On-policy) |
| 样本效率 | 高 | 中等 |
| 实时性能 | 较好 | 好 |
| 训练时间 | 较短 | 较长 |
| 避障策略 | LLM + 几何 | LLM + 几何 |
| 连续导航 | ✅ 支持 | ✅ 支持 |
| 网络控制 | ✅ 支持 | ✅ 支持 |
| 轨迹记录 | ✅ 支持 | ✅ 支持 |

## 完整工作流程示例

### 1. 训练 SAC 模型

```bash
# 激活环境
source myenv/bin/activate

# 训练单智能体模型（推荐）
python -m gym_pybullet_drones.custom.sac_learn \
    --local true \
    --gui false

# 等待训练完成（约 30-60 分钟，取决于硬件）
# 模型保存在: results/sac-save-<timestamp>/best_model.zip
```

### 2. 测试单目标避障

```bash
# 使用 LLM 避障
export OPENAI_API_KEY='your-api-key-here'
python -m gym_pybullet_drones.custom.sac_llm_obstacle_avoidance \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm true

# 仅几何避障
python -m gym_pybullet_drones.custom.sac_llm_obstacle_avoidance \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm false
```

### 3. 连续导航（多目标）

```bash
# 启动连续导航系统
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm true

# 通过网络发送目标点
python -c "
import socket, json
cmd = {'type': 'target', 'target': [1.0, 1.0, 1.0]}
s = socket.socket()
s.connect(('localhost', 8888))
s.send(json.dumps(cmd).encode())
print(s.recv(1024).decode())
s.close()
"
```

### 4. 查看训练日志

```bash
# 启动 TensorBoard
tensorboard --logdir results/sac-save-<timestamp>/tb/

# 在浏览器打开 http://localhost:6006
```

## 许可证

与主项目相同（MIT License）
