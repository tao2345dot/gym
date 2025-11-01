# SAC 连续导航障碍物配置更新

## 更新内容

已将 SAC 多障碍物连续导航环境中的障碍物更改为：

### ✅ 新配置
- **4 根固定大小的柱子**
- **随机位置生成**（每次启动不同）
- **统一规格**：半径 0.15m，高度 2.0m
- **灰色外观**：更专业的工业风格

### 📋 柱子参数

```python
PILLAR_CONFIG = {
    'radius': 0.15,      # 固定半径 15cm
    'height': 2.0,       # 固定高度 2m
    'count': 4,          # 数量 4 根
    'color': [0.6, 0.6, 0.6, 1.0],  # 灰色
}
```

### 🎲 随机生成规则

每次启动系统时，柱子位置会随机生成，但满足以下约束：

1. **位置范围**：
   - X: [-1.5m, +1.5m]
   - Y: [-1.5m, +1.5m]
   - Z: 1.0m (柱子中心高度)

2. **安全约束**：
   - 原点安全半径：0.5m（避免柱子生成在起点）
   - 柱子间最小距离：0.8m（避免柱子重叠）

3. **生成策略**：
   - 最多尝试 100 次放置每根柱子
   - 如果无法满足约束，会输出警告

## 优势

### 🎯 训练优势
- **固定尺寸**：简化学习难度，模型更容易泛化
- **随机位置**：每次训练环境不同，提高鲁棒性
- **合理约束**：保证环境可导航，避免无解情况

### 🏗️ 工程优势
- **易于配置**：只需修改 `PILLAR_CONFIG`
- **可视化清晰**：灰色柱子在仿真中醒目
- **性能优化**：4 根柱子提供足够挑战且不影响性能

## 修改的文件

1. **核心代码**
   - `sac_continuous_navigator.py` - 添加随机生成函数，修改障碍物管理
   - `test_sac_continuous.py` - 更新测试脚本的障碍物配置

2. **文档更新**
   - `SAC_README.md` - 更新障碍物配置说明
   - `SAC_CONTINUOUS_GUIDE.md` - 详细配置指南
   - `SAC_CONTINUOUS_SUMMARY.md` - 总结文档

## 使用示例

### 运行测试
```bash
# 快速测试（无需训练模型）
python -m gym_pybullet_drones.custom.test_sac_continuous
```

每次运行都会看到不同的柱子布局：
```
[障碍物] 正在生成随机柱子...
[障碍物] ✅ 成功生成 4 根柱子:
  柱子 1: 位置=(-0.85, 1.23), 半径=0.15m, 高度=2.00m
  柱子 2: 位置=(1.15, -0.67), 半径=0.15m, 高度=2.00m
  柱子 3: 位置=(-1.32, -0.92), 半径=0.15m, 高度=2.00m
  柱子 4: 位置=(0.78, 1.05), 半径=0.15m, 高度=2.00m
```

### 完整导航
```bash
# 启动连续导航（使用训练好的模型）
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-<timestamp>/best_model.zip \
    --gui true \
    --use_llm true
```

## 自定义配置

### 修改柱子数量
```python
PILLAR_CONFIG = {
    'count': 6,  # 增加到 6 根
    # ... 其他参数
}
```

### 修改柱子大小
```python
PILLAR_CONFIG = {
    'radius': 0.2,   # 更粗的柱子
    'height': 3.0,   # 更高的柱子
    # ... 其他参数
}
```

### 调整生成范围
在 `_generate_obstacles` 方法中：
```python
self.obstacles = generate_random_pillars(
    count=4,
    x_range=(-2.0, 2.0),  # 更大的范围
    y_range=(-2.0, 2.0),
    min_distance=1.0,     # 更大的间距
    origin_safe_radius=0.8  # 更大的安全区
)
```

### 使用固定位置
如需固定位置（用于重复测试），直接赋值：
```python
def _generate_obstacles(self):
    self.obstacles = [
        {'pos': [1.0, 1.0, 1.0], 'radius': 0.15, 'height': 2.0},
        {'pos': [1.0, -1.0, 1.0], 'radius': 0.15, 'height': 2.0},
        {'pos': [-1.0, 1.0, 1.0], 'radius': 0.15, 'height': 2.0},
        {'pos': [-1.0, -1.0, 1.0], 'radius': 0.15, 'height': 2.0},
    ]
    print(f"[障碍物] 使用固定柱子位置（测试模式）")
```

## 可视化效果

### 仿真窗口
- 4 根**灰色圆柱体**分布在环境中
- 位置每次启动随机生成
- 绿色球体标记目标点
- 蓝色球体标记避障路径点（如有）

### 轨迹图
运行结束后自动生成 `sac_continuous_navigation_trajectory.png`：
- **3D 图**：显示飞行轨迹和柱子位置
- **XY 投影**：俯视图显示柱子圆形投影（灰色圆圈）
- **高度图**：Z 轴随时间变化

## 技术细节

### 碰撞检测
```python
def check_collision_risk(self, current_pos):
    for obs in self.obstacles:  # 使用动态生成的柱子列表
        distance_xy = np.linalg.norm(current_pos[:2] - obs_pos[:2])
        if distance_xy < (obs['radius'] + COLLISION_THRESHOLD):
            # 检测到碰撞风险
            return True, obs
    return False, None
```

### 避障规划
- **LLM 模式**：将柱子信息传递给 OpenAI API 进行智能规划
- **几何模式**：计算垂直于柱子-无人机连线的避障点

## 注意事项

⚠️ **训练模型时的障碍物**：
- 如果使用已训练的模型，确保训练时的障碍物配置与测试相似
- 固定大小的柱子有助于模型泛化到不同位置
- 建议使用相同的 `radius` 和 `height` 参数

✅ **最佳实践**：
- 训练时也使用随机位置（提高泛化能力）
- 测试时可以用固定位置（方便对比不同算法）
- 记录柱子位置到日志（便于复现和分析）

## 向后兼容

如需恢复旧的 3 个不同大小障碍物配置：
```python
# 在 _generate_obstacles 方法中
self.obstacles = [
    {'pos': [0.5, 0.5, 0.5], 'radius': 0.3, 'height': 1.0},
    {'pos': [-0.5, 0.5, 0.5], 'radius': 0.25, 'height': 0.8},
    {'pos': [0.0, -0.5, 0.5], 'radius': 0.35, 'height': 1.2},
]
```

---

✅ **更新已完成，系统可立即使用！**
