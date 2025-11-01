# SAC 连续导航使用指南

## 快速开始

### 1. 准备环境

确保已激活虚拟环境并安装所有依赖：

```bash
source myenv/bin/activate
```

### 2. 训练 SAC 模型

首先需要训练一个 SAC 模型：

```bash
# 训练单智能体模型（推荐）
python -m gym_pybullet_drones.custom.sac_learn --local true --gui false

# 训练过程约 30-60 分钟
# 模型保存位置: results/sac-save-<timestamp>/best_model.zip
```

训练完成后，记下模型路径，例如：
```
results/sac-save-05.01.2025_14.30.45/best_model.zip
```

### 3. 快速测试（可选）

在没有训练好的模型时，可以先运行快速测试验证系统功能：

```bash
python -m gym_pybullet_drones.custom.test_sac_continuous
```

注意：快速测试使用随机策略，性能有限，仅用于验证环境和障碍物设置。

### 4. 运行连续导航

使用训练好的模型启动连续导航：

```bash
# 使用 LLM 避障（需要 OpenAI API Key）
export OPENAI_API_KEY='your-api-key-here'
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm true

# 或者仅使用几何避障（无需 API Key）
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm false
```

## 控制方法

### 键盘控制

在 GUI 窗口激活状态下：

- **空格键**: 暂停/继续导航
- **H 键**: 返回起点（添加起点到目标队列）
- **Q 键 / ESC**: 退出程序

### 网络控制

通过 TCP 端口 8888 发送 JSON 命令：

#### Python 示例

```python
import socket
import json

def send_target(x, y, z):
    """发送目标点"""
    command = {
        'type': 'target',
        'target': [x, y, z]
    }
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 8888))
    sock.send(json.dumps(command).encode('utf-8'))
    response = sock.recv(1024)
    sock.close()
    
    return json.loads(response.decode('utf-8'))

# 发送多个目标点实现连续导航
send_target(1.0, 1.0, 1.0)
send_target(1.0, -1.0, 1.0)
send_target(-1.0, -1.0, 0.5)
send_target(0.0, 0.0, 1.5)
```

#### Bash 示例

```bash
# 使用 netcat 发送命令
echo '{"type":"target","target":[1.0,1.0,1.0]}' | nc localhost 8888

# 使用 Python 一行命令
python -c "
import socket, json
s = socket.socket()
s.connect(('localhost', 8888))
s.send(json.dumps({'type': 'target', 'target': [1.0, 1.0, 1.0]}).encode())
print(s.recv(1024).decode())
s.close()
"
```

## 配置参数

### 障碍物配置

系统使用 **4 根固定大小、随机位置的柱子**作为障碍物。在 `sac_continuous_navigator.py` 中修改柱子配置：

```python
# 柱子固定配置
PILLAR_CONFIG = {
    'radius': 0.08,      # 柱子半径（米）
    'height': 2.0,       # 柱子高度（米）
    'count': 4,          # 柱子数量
    'color': [0.6, 0.6, 0.6, 1.0],  # 灰色
}
```

**随机生成参数：**

```python
# 在 generate_random_pillars 函数中调整
x_range=(-1.5, 1.5)           # X 坐标范围
y_range=(-1.5, 1.5)           # Y 坐标范围
min_distance=0.8              # 柱子之间最小距离（米）
origin_safe_radius=0.5        # 原点安全半径（米）
```

**特点：**
- 柱子位置每次启动时随机生成
- 保证不会生成在起点附近（安全半径 0.5m）
- 柱子之间保持最小距离（0.8m）
- 固定的柱子尺寸便于训练和测试

### 避障参数

```python
# 碰撞检测距离阈值（障碍物半径 + 安全余量）
COLLISION_THRESHOLD = 0.5  

# 路径点到达阈值（米）
WAYPOINT_REACH_THRESHOLD = 0.15

# 目标点到达阈值（米）
TARGET_REACH_THRESHOLD = 0.2
```

### 扩展空间配置

在 `config_continuous.py` 中修改：

```python
EXTENDED_SPACE_CONFIG = {
    'x_min': -2.0,
    'x_max': 2.0,
    'y_min': -2.0,
    'y_max': 2.0,
    'z_min': 0.1,
    'z_max': 2.5
}
```

## 工作流程

### 系统启动流程

```
启动程序
    ↓
加载 SAC 模型
    ↓
创建扩展环境
    ↓
添加障碍物和标记
    ↓
初始化控制器
    ↓
启动网络服务器
    ↓
进入主循环
```

### 导航流程

```
获取当前位置
    ↓
检查目标队列 ───→ 队列为空？ ───→ 等待新目标
    ↓ 队列非空
取出下一个目标
    ↓
检测碰撞风险？
    ├─ 是 → 进入避障模式
    │       ├─ 使用 LLM 规划路径点（如果启用）
    │       │  失败 ↓
    │       └─ 使用几何规划路径点
    │       ↓
    │   导航到避障路径点
    │       ↓
    │   到达路径点？
    │       ├─ 是 → 退出避障模式
    │       └─ 否 → 继续导航
    │
    └─ 否 → 直接导航到目标
            ↓
        使用 SAC 模型推理动作
            ↓
        执行动作并更新环境
            ↓
        到达目标？
            ├─ 是 → 记录轨迹，取下一目标
            └─ 否 → 继续导航
```

## 避障策略

### LLM 避障

使用 OpenAI GPT 模型进行智能路径规划：

**优点：**
- 考虑多个障碍物的全局影响
- 可生成更优的路径
- 适应复杂环境

**缺点：**
- 需要 API Key 和网络连接
- 有一定延迟（~1-2秒）
- 需要消耗 API 额度

**使用条件：**
```bash
export OPENAI_API_KEY='sk-...'
python -m ... --use_llm true
```

### 几何避障

使用简单的几何规划算法：

**优点：**
- 无需网络和 API
- 实时计算，无延迟
- 完全免费

**缺点：**
- 仅考虑单个最近障碍物
- 路径可能不是最优
- 复杂环境下可能失效

**使用方法：**
```bash
python -m ... --use_llm false
```

## 输出和可视化

### 实时显示

程序运行时会实时显示：

```
[导航] Step 150: Pos=[ 0.45 -0.32  0.87], Target=[1.0 1.0 1.0], Dist=0.85m
[避障] ⚠️ 检测到障碍物，进入避障模式
[避障] 使用几何规划路径点: [0.2 0.5 0.87]
[避障] ✅ 到达避障路径点，恢复目标导航
[导航] ✅ 到达目标 #1: [1.0, 1.0, 1.0]
```

### 统计信息

程序退出时会显示：

```
📊 运行统计
============================================================
运行时长: 125.34 秒
到达目标数: 4
总步数: 6267
避障次数: 7
处理命令数: 4
============================================================
```

### 轨迹可视化

程序会自动生成轨迹图并保存为 `sac_continuous_navigation_trajectory.png`，包含：

1. **3D 飞行轨迹图**
   - 蓝色线条：实际飞行轨迹
   - 绿色星号：到达的目标点
   - 红色圆点：障碍物位置

2. **XY 平面投影**
   - 俯视图显示水平面运动
   - 红色圆圈：障碍物投影

3. **高度随时间变化**
   - 显示垂直方向的控制效果

## 故障排除

### 问题 1: 模型加载失败

```
❌ 模型加载失败: [Errno 2] No such file or directory: 'xxx.zip'
```

**解决方法：**
- 确认模型路径正确
- 使用绝对路径或相对于运行目录的路径
- 先训练模型再运行

### 问题 2: LLM 避障失败

```
[避障] LLM 规划失败: AuthenticationError, 使用几何规划
```

**解决方法：**
- 检查 `OPENAI_API_KEY` 是否正确设置
- 确认 API Key 有效且有额度
- 或使用 `--use_llm false` 禁用 LLM

### 问题 3: 无人机碰撞障碍物

**可能原因：**
- 避障阈值设置太小
- SAC 模型训练不充分
- 障碍物太密集

**解决方法：**
- 增大 `COLLISION_THRESHOLD`
- 延长训练时间
- 减少障碍物或增大间距
- 降低导航速度（修改控制频率）

### 问题 4: 网络命令发送失败

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**解决方法：**
- 确认导航系统已启动
- 检查端口 8888 是否被占用
- 防火墙是否允许本地连接

## 进阶使用

### 自定义柱子布局

如果需要固定的柱子位置而非随机生成，可以修改 `_generate_obstacles` 方法：

```python
# 在 SACContinuousNavigator 类中
def _generate_obstacles(self):
    """生成固定位置的柱子（用于特定测试场景）"""
    print(f"[障碍物] 使用固定柱子位置...")
    
    self.obstacles = [
        # 四角布局
        {'pos': [1.0, 1.0, 1.0], 'radius': 0.15, 'height': 2.0},
        {'pos': [1.0, -1.0, 1.0], 'radius': 0.15, 'height': 2.0},
        {'pos': [-1.0, 1.0, 1.0], 'radius': 0.15, 'height': 2.0},
        {'pos': [-1.0, -1.0, 1.0], 'radius': 0.15, 'height': 2.0},
    ]
```

或调整随机生成的参数范围：

```python
# 更密集的柱子布局
self.obstacles = generate_random_pillars(
    count=6,                    # 增加到 6 根
    radius=0.15,
    height=2.0,
    x_range=(-1.0, 1.0),       # 缩小范围
    y_range=(-1.0, 1.0),
    min_distance=0.6,          # 减小最小距离
    origin_safe_radius=0.3
)
```

### 动态目标序列

编写脚本自动发送目标序列：

```python
# auto_mission.py
import time
from send_target import send_target

# 定义巡航路径
waypoints = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [0.0, 0.0, 1.5],  # 返回中心并上升
]

# 按顺序发送
for i, wp in enumerate(waypoints):
    print(f"发送路径点 {i+1}/{len(waypoints)}: {wp}")
    response = send_target(*wp)
    print(f"  响应: {response}")
    time.sleep(2)  # 等待 2 秒

print("任务序列发送完成")
```

### 集成自定义传感器

在环境中添加额外传感器（如激光雷达）：

```python
# 在 ExtendedHoverAviary 中添加
def _computeObs(self):
    obs = super()._computeObs()
    
    # 添加激光雷达扫描
    lidar_data = self._getLidarScan()
    
    # 拼接到观测中
    return np.hstack([obs, lidar_data])
```

## 性能优化建议

1. **训练优化**
   - 增加训练步数获得更好性能
   - 使用 GPU 加速训练
   - 调整超参数（学习率、批量大小等）

2. **实时性优化**
   - 降低 LLM 调用频率
   - 使用异步 LLM 调用
   - 缓存避障路径结果

3. **稳定性优化**
   - 增加碰撞检测频率
   - 平滑路径规划结果
   - 添加安全高度限制

## 相关文件

- `sac_continuous_navigator.py`: 主导航控制器
- `start_sac_continuous.py`: 启动脚本
- `test_sac_continuous.py`: 快速测试脚本
- `sac_learn.py`: SAC 训练脚本
- `sac_llm_obstacle_avoidance.py`: 单目标避障示例
- `config_continuous.py`: 配置文件
- `space_expander.py`: 扩展环境定义

## 更多资源

- [SAC 算法论文](https://arxiv.org/abs/1801.01290)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [gym-pybullet-drones 项目](https://github.com/utiasDSL/gym-pybullet-drones)
