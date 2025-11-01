# SAC 远程控制器使用指南

## 📖 简介

`sac_remote_controller.py` 是 SAC 连续导航系统的独立远程控制器，允许你在单独的终端中通过键盘命令控制无人机，而不会被导航系统的日志输出干扰。

## 🚀 快速开始

### 1. 启动导航系统（终端 1）

```bash
source myenv/bin/activate
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-11.01.2025_14.14.27/best_model.zip \
    --gui true \
    --use_llm false
```

### 2. 启动远程控制器（终端 2）

```bash
source myenv/bin/activate
python -m gym_pybullet_drones.custom.sac_remote_controller
```

### 3. 发送命令

控制器启动后，你会看到提示符 `🎮 SAC>`，可以输入以下命令：

#### 基础命令

```bash
# 显示帮助
help

# 设置目标点 (x y z)
0.5 0.5 0.3

# 返回起点
home

# 检查连接状态
status

# 退出控制器
exit
```

#### 快速测试命令

```bash
# 前往预定义测试点
test1    # 前往 [0.5, 0.5, 0.3]
test2    # 前往 [0.8, 0.3, 0.3]
test3    # 前往 [0.3, 0.8, 0.3]
test4    # 前往 [-0.5, 0.5, 0.3]
test5    # 前往 [-0.5, -0.5, 0.3]

# 依次访问所有测试点
testall
```

#### 预定义任务

```bash
# 正方形巡航
square

# 三角形巡航
triangle

# Z字形巡航
zigzag

# 螺旋上升
spiral
```

## 📋 命令详解

### 目标点命令

**格式**: `x y z`

**示例**:
```bash
🎮 SAC> 0.5 0.5 0.3
🎯 设置测试点: [0.5, 0.5, 0.3]
✅ 目标已添加: [0.5, 0.5, 0.3]
```

**坐标范围建议**:
- X/Y: `[-0.8, 0.8]` (训练空间: [-1.5, 1.5])
- Z: `[0.2, 0.5]` (训练空间: [0.05, 2.5])

> ⚠️ 超出建议范围会收到警告，但仍可继续执行

### 返回起点

```bash
🎮 SAC> home
🏠 返回起点...
✅ 返回起点命令已发送
```

### 预定义任务示例

#### 正方形巡航
```bash
🎮 SAC> square
🔲 正方形巡航任务...
   ✓ 航点 1/5: [0.5, 0.5, 0.3]
   ✓ 航点 2/5: [0.5, -0.5, 0.3]
   ✓ 航点 3/5: [-0.5, -0.5, 0.3]
   ✓ 航点 4/5: [-0.5, 0.5, 0.3]
   ✓ 航点 5/5: [0.0, 0.0, 0.3]
```

## 🔧 高级用法

### 自定义主机和端口

```bash
# 连接到自定义地址
python -m gym_pybullet_drones.custom.sac_remote_controller \
    --host 192.168.1.100 \
    --port 8888
```

### 在 Python 脚本中使用

```python
from gym_pybullet_drones.custom.sac_remote_controller import SACRemoteController

# 创建控制器
controller = SACRemoteController(host='localhost', port=8888)

# 发送目标点
controller.send_command('target', target=[0.5, 0.5, 0.3])

# 返回起点
controller.send_command('home')
```

## 📊 完整工作流示例

### 示例 1: 简单导航测试

**终端 1 (导航系统)**:
```bash
python -m gym_pybullet_drones.custom.start_sac_continuous \
    --model_path results/sac-save-11.01.2025_14.14.27/best_model.zip \
    --gui true \
    --use_llm false
```

**终端 2 (远程控制器)**:
```bash
python -m gym_pybullet_drones.custom.sac_remote_controller

# 等待看到提示符，然后输入:
🎮 SAC> test1
🎮 SAC> test2
🎮 SAC> test3
🎮 SAC> home
```

### 示例 2: 正方形巡航

**终端 2**:
```bash
🎮 SAC> square
# 无人机将自动按正方形路径飞行
```

### 示例 3: 自定义路径

**终端 2**:
```bash
🎮 SAC> 0.5 0.0 0.3
🎮 SAC> 0.5 0.5 0.3
🎮 SAC> 0.0 0.5 0.3
🎮 SAC> 0.0 0.0 0.3
```

## ❓ 常见问题

### Q1: 连接失败？

**问题**: `❌ 无法连接到导航系统 localhost:8888`

**解决**:
1. 确认导航系统正在运行（终端 1）
2. 检查端口是否正确（默认 8888）
3. 使用 `status` 命令测试连接

### Q2: 命令发送后无响应？

**可能原因**:
- 无人机正在前往当前目标
- 检查终端 1 的导航系统日志
- 使用导航系统的键盘控制（空格键暂停/继续）

### Q3: 坐标范围警告？

如果输入的坐标超出建议范围，会收到警告：
```bash
🎮 SAC> 2.0 2.0 0.3
⚠️  警告: X/Y 超出训练范围 [-1.5, 1.5]
   当前: X=2.00, Y=2.00
   是否继续? (y/n): 
```

输入 `y` 继续或 `n` 取消。

## 🎯 最佳实践

1. **先测试低高度**: 从 Z=0.3m 开始，成功后再增加高度
2. **小步前进**: 先发送近距离目标，观察无人机行为
3. **使用测试点**: `test1-5` 是经过验证的安全目标
4. **监控终端 1**: 注意导航系统的日志输出
5. **避障感知**: 注意环境中的 4 根灰色柱子

## 📚 相关文档

- `SAC_README.md` - SAC 系统总览
- `SAC_CONTINUOUS_GUIDE.md` - 详细使用指南  
- `SAC_CONTINUOUS_SUMMARY.md` - 完整实现总结

## 🆚 vs PPO 远程控制器

| 特性 | SAC Controller | PPO Controller |
|------|---------------|----------------|
| 默认端口 | 8888 | 12345 |
| 目标高度 | 0.3m (更低) | 1.0m+ |
| LLM 圆形轨迹 | 半径 0.8m | 半径 2.5m |
| 空间范围 | [-1.5, 1.5] | [-4.0, 4.0] |
| 适用场景 | 小范围精确导航 | 大范围探索 |

---

✅ **现在你可以在单独的终端中方便地控制 SAC 导航系统了！**
