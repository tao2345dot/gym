"""
LLM圆形轨迹测试和集成脚本

测试LLM生成的圆形轨迹，并展示如何与现有导航系统集成

使用方法:
    1. 首先在 llm_circle_planner.py 中填写API密钥
    2. 运行此脚本进行测试: python test_llm_trajectory.py
    3. 成功后可集成到连续导航系统中
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Ubuntu 24中文字体配置 - 使用您成功的方法
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
chinese_font = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from gym_pybullet_drones.custom.llm_circle_planner import generate_circle_trajectory
    print("✅ 成功导入 LLM轨迹规划器")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装 openai 包: pip install openai")
    sys.exit(1)


def visualize_trajectory(trajectory, title="LLM生成的圆形轨迹"):
    """
    可视化3D轨迹
    
    参数:
        trajectory: 形状为 (1, N, 3) 的轨迹数组
        title: 图表标题
    """
    if trajectory is None:
        print("❌ 轨迹为空，无法可视化")
        return
    
    fig = plt.figure(figsize=(12, 4))
    
    # 提取单无人机轨迹
    traj = trajectory[0]  # 形状: (N, 3)
    
    # 3D轨迹图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2)
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, label='起点')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, label='终点')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹', fontproperties=chinese_font)
    ax1.legend(prop=chinese_font)
    
    # XY平面投影
    ax2 = fig.add_subplot(132)
    ax2.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
    ax2.scatter(traj[0, 0], traj[0, 1], c='green', s=100, label='起点')
    ax2.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, label='终点')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY平面投影 (圆形轨迹)', fontproperties=chinese_font)
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend(prop=chinese_font)
    
    # 高度变化
    ax3 = fig.add_subplot(133)
    ax3.plot(range(len(traj)), traj[:, 2], 'g-', linewidth=2)
    ax3.set_xlabel('轨迹点索引', fontproperties=chinese_font)
    ax3.set_ylabel('Z (m)')
    ax3.set_title('高度变化', fontproperties=chinese_font)
    ax3.grid(True)
    
    # 使用您成功的字体配置方法
    plt.suptitle(title, fontsize=14, fontproperties=chinese_font)
    plt.tight_layout()
    plt.show()


def test_different_configurations():
    """
    测试不同配置的轨迹生成
    """
    print("\n" + "="*60)
    print("🧪 测试不同配置的轨迹生成")
    print("="*60)
    
    test_cases = [
        {
            "name": "标准测试 - 逆时针",
            "position": [2.0, 0.0, 1.5],
            "waypoints": 50,
            "clockwise": False
        },
        {
            "name": "顺时针测试",
            "position": [0.0, 2.0, 2.0],
            "waypoints": 50,
            "clockwise": True
        },
        {
            "name": "不同起始位置",
            "position": [1.5, 1.5, 1.0],
            "waypoints": 50,
            "clockwise": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {test_case['name']}")
        print(f"   位置: {test_case['position']}")
        print(f"   点数: {test_case['waypoints']}")
        print(f"   方向: {'顺时针' if test_case['clockwise'] else '逆时针'}")
        
        trajectory = generate_circle_trajectory(
            init_xyz=test_case["position"],
            num_waypoints=test_case["waypoints"],
            clockwise=test_case["clockwise"]
        )
        
        if trajectory is not None:
            print(f"   ✅ 成功生成轨迹: {trajectory.shape}")
            
            # 验证圆形性质
            points_2d = trajectory[0, :, :2]
            distances = np.sqrt(np.sum(points_2d**2, axis=1))
            radius_std = np.std(distances)
            
            print(f"   📊 半径标准差: {radius_std:.6f} (越小越圆)")
            
            if i == 1:  # 只为第一个测试显示可视化
                visualize_trajectory(trajectory, f"测试 {i}: {test_case['name']}")
        else:
            print(f"   ❌ 轨迹生成失败")


def check_api_setup():
    """
    检查API配置是否正确
    """
    print("🔧 检查API配置...")
    
    from gym_pybullet_drones.custom.llm_circle_planner import generate_circle_trajectory
    import inspect
    
    # 读取源代码检查API配置
    source_lines = inspect.getsource(generate_circle_trajectory)
    
    if 'api_key=""' in source_lines:
        print("⚠️  警告: API Key 还未填写")
        print("   请在 llm_circle_planner.py 第41行左右填写您的硅基流动API Key")
        return False
        
    if 'base_url=""' in source_lines:
        print("⚠️  警告: Base URL 还未填写")
        print("   请在 llm_circle_planner.py 第42行左右填写硅基流动的Base URL")
        return False
    
    print("✅ API配置检查通过")
    return True


def integration_example():
    """
    展示如何与现有导航系统集成的示例
    """
    print("\n" + "="*60)
    print("🔗 集成示例: 如何在连续导航中使用LLM轨迹")
    print("="*60)
    
    print("""
集成步骤:

1. 在 continuous_navigator.py 中导入:
   from gym_pybullet_drones.custom.llm_circle_planner import generate_circle_trajectory

2. 在目标设定函数中调用:
   def set_circle_mission(self, center=[0,0], radius=2.0, height=1.5):
       init_pos = [radius, 0, height]  # 起始位置
       trajectory = generate_circle_trajectory(
           init_xyz=init_pos,
           num_waypoints=1000,
           clockwise=False
       )
       if trajectory is not None:
           # 将轨迹点逐个添加到目标队列
           for point in trajectory[0]:
               self.add_target(point)

3. 在键盘控制中添加快捷键:
   - 按 'C' 键: 生成圆形任务
   - 按 'R' 键: 反向圆形任务

这样就可以实现LLM驱动的智能导航了！
    """)


def main():
    """
    主测试流程
    """
    print("🚁 LLM圆形轨迹生成器 - 综合测试")
    print("=" * 60)
    
    # 1. 检查API配置
    if not check_api_setup():
        print("\n💡 请先配置API密钥后再运行测试")
        return
    
    # 2. 基础功能测试
    print("\n📋 开始基础功能测试...")
    test_position = [2.0, 0.0, 1.5]
    trajectory = generate_circle_trajectory(
        init_xyz=test_position,
        num_waypoints=100,
        clockwise=False
    )
    
    if trajectory is not None:
        print("✅ 基础测试通过!")
        
        # 3. 可视化展示
        try:
            visualize_trajectory(trajectory, "LLM生成的圆形轨迹 - 基础测试")
        except Exception as e:
            print(f"⚠️  可视化跳过 (需要matplotlib): {e}")
        
        # 4. 多配置测试
        test_different_configurations()
        
        # 5. 集成说明
        integration_example()
        
        print("\n🎉 所有测试完成!")
        
    else:
        print("❌ 基础测试失败，请检查API配置和网络连接")


if __name__ == "__main__":
    main()
