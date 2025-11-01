#!/usr/bin/env python3
"""
SAC 远程控制器快速测试

测试与导航系统的连接和基本命令
"""

from gym_pybullet_drones.custom.sac_remote_controller import SACRemoteController
import time

def test_connection():
    """测试连接"""
    print("=" * 60)
    print("测试 1: 连接测试")
    print("=" * 60)
    
    controller = SACRemoteController()
    
    # 尝试发送状态命令
    print("尝试连接到 localhost:8888...")
    result = controller.send_command('home')
    
    if result:
        print("✅ 连接成功！")
        return True
    else:
        print("❌ 连接失败")
        print("\n请确保导航系统正在运行:")
        print("  python -m gym_pybullet_drones.custom.start_sac_continuous \\")
        print("      --model_path results/sac-save-<timestamp>/best_model.zip \\")
        print("      --gui true --use_llm false")
        return False

def test_commands():
    """测试基本命令"""
    print("\n" + "=" * 60)
    print("测试 2: 基本命令测试")
    print("=" * 60)
    
    controller = SACRemoteController()
    
    test_cases = [
        ("返回起点", 'home', None),
        ("发送目标点 1", 'target', [0.3, 0.3, 0.3]),
        ("发送目标点 2", 'target', [0.5, 0.0, 0.3]),
        ("再次返回起点", 'home', None),
    ]
    
    success_count = 0
    
    for description, cmd_type, target in test_cases:
        print(f"\n📝 {description}...")
        if controller.send_command(cmd_type, target):
            success_count += 1
            print(f"   ✅ 成功")
        else:
            print(f"   ❌ 失败")
        time.sleep(0.5)
    
    print(f"\n结果: {success_count}/{len(test_cases)} 个命令成功")
    return success_count == len(test_cases)

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  🧪 SAC 远程控制器测试")
    print("=" * 60 + "\n")
    
    # 测试连接
    if not test_connection():
        print("\n❌ 连接测试失败，跳过后续测试")
        return
    
    # 等待用户确认
    print("\n⏸  请确保导航系统已启动并准备就绪")
    input("按 Enter 继续测试基本命令...")
    
    # 测试命令
    test_commands()
    
    print("\n" + "=" * 60)
    print("  ✅ 测试完成")
    print("=" * 60)
    print("\n💡 提示:")
    print("  1. 如果测试成功，可以运行完整的远程控制器:")
    print("     python -m gym_pybullet_drones.custom.sac_remote_controller")
    print("\n  2. 或查看使用指南:")
    print("     cat gym_pybullet_drones/custom/SAC_REMOTE_CONTROLLER_GUIDE.md")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 测试被中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
