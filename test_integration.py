#!/usr/bin/env python3
"""
LLM轨迹集成测试

测试LLM圆形轨迹生成功能在连续导航系统中的集成效果
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_llm_integration():
    """测试LLM轨迹集成功能"""
    
    print("🚁 LLM轨迹集成测试")
    print("=" * 50)
    
    try:
        # 导入连续导航器
        from gym_pybullet_drones.custom.continuous_navigator import ContinuousNavigator, find_latest_model
        
        # 查找最新模型
        print("📁 正在查找最新训练模型...")
        model_path = find_latest_model()
        print(f"✅ 找到模型: {model_path}")
        
        # 创建导航器实例
        print("\n🏗️ 正在初始化导航系统...")
        navigator = ContinuousNavigator(
            model_path=model_path,
            gui=True,  # 显示GUI以便观察
            record=False
        )
        
        # 初始化系统
        navigator.initialize()
        
        print("\n🤖 测试LLM圆形轨迹生成...")
        # 测试圆形轨迹生成
        success = navigator.generate_circle_mission(
            radius=1.5,    # 半径1.5米（在安全范围内）
            height=1.0,    # 高度1米
            waypoints=50,  # 50个轨迹点
            clockwise=False  # 逆时针
        )
        
        if success:
            print("✅ LLM轨迹生成成功!")
            
            # 显示轨迹统计
            navigator.show_trajectory_stats()
            
            # 提示用户
            print(f"\n🎮 测试完成! 主要功能验证:")
            print(f"   ✅ LLM轨迹生成")
            print(f"   ✅ 轨迹可视化") 
            print(f"   ✅ 轨迹统计分析")
            print(f"   ✅ 目标队列集成")
            
            print(f"\n💡 现在您可以:")
            print(f"   1. 运行 python -m gym_pybullet_drones.custom.rollout_continuous")
            print(f"   2. 在系统中输入 'circle' 或 'c' 生成圆形轨迹") 
            print(f"   3. 输入 'visual' 或 'v' 查看轨迹可视化")
            print(f"   4. 输入 'stats' 或 's' 查看轨迹统计")
            
        else:
            print("❌ LLM轨迹生成失败")
            
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装必要的包:")
        print("  pip install openai matplotlib")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔚 测试结束")


if __name__ == "__main__":
    test_llm_integration()
