"""
SAC 连续导航启动脚本

快速启动 SAC 连续导航系统的便捷脚本
"""

import os
import sys

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
except Exception:
    pass

from gym_pybullet_drones.custom.sac_continuous_navigator import SACContinuousNavigator

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC 连续导航系统 - 快速启动')
    parser.add_argument('--model_path', type=str, required=True,
                       help='SAC 模型路径 (必需)')
    parser.add_argument('--gui', type=str, default='true',
                       help='是否显示 GUI (默认: true)')
    parser.add_argument('--record', type=str, default='false',
                       help='是否录制视频 (默认: false)')
    parser.add_argument('--use_llm', type=str, default='true',
                       help='是否使用 LLM 避障 (默认: true)')
    
    args = parser.parse_args()
    
    # 转换字符串参数为布尔值
    from gym_pybullet_drones.utils.utils import str2bool
    gui = str2bool(args.gui)
    record = str2bool(args.record)
    use_llm = str2bool(args.use_llm)
    
    print("\n" + "="*70)
    print("🚁 SAC 连续导航系统")
    print("="*70)
    print(f"模型路径: {args.model_path}")
    print(f"GUI 模式: {'✅ 开启' if gui else '❌ 关闭'}")
    print(f"录制视频: {'✅ 开启' if record else '❌ 关闭'}")
    print(f"LLM 避障: {'✅ 开启' if use_llm else '❌ 关闭 (使用几何避障)'}")
    print("="*70)
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件不存在: {args.model_path}")
        print("\n请先训练模型:")
        print("  python -m gym_pybullet_drones.custom.sac_learn --local true --gui false")
        return
    
    # 检查 LLM API Key (如果需要)
    if use_llm:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("\n⚠️ 警告: 未设置 OPENAI_API_KEY 环境变量")
            print("将使用几何避障规划作为备选")
            print("\n如需使用 LLM 避障，请设置:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
        else:
            print(f"✅ 已检测到 OPENAI_API_KEY")
    
    print("\n" + "="*70)
    print("控制说明:")
    print("  空格键: 暂停/继续")
    print("  H 键:   返回起点")
    print("  Q/ESC:  退出程序")
    print("  网络命令: 通过 TCP 端口 8888 发送目标")
    print("="*70 + "\n")
    
    try:
        # 创建导航器
        navigator = SACContinuousNavigator(
            model_path=args.model_path,
            gui=gui,
            record=record,
            use_llm=use_llm
        )
        
        # 初始化系统
        navigator.initialize()
        
        # 启动导航
        navigator.start_navigation()
        
    except KeyboardInterrupt:
        print("\n[系统] 接收到中断信号，正在退出...")
    except Exception as e:
        print(f"\n[系统] ❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
