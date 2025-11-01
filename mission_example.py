"""
批量任务执行示例

展示如何自动发送多个目标点，实现自动巡航任务
"""

import time
import sys
import os

# 添加路径以导入 send_command
sys.path.append(os.path.dirname(__file__))
from send_command import send_command


def execute_mission(waypoints, host='localhost', port=8888, delay=2.0):
    """
    执行巡航任务
    
    参数:
        waypoints: 路径点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        host: 服务器地址
        port: 服务器端口
        delay: 发送间隔（秒）
    """
    print("\n" + "="*70)
    print("🚁 SAC 连续导航 - 自动巡航任务")
    print("="*70)
    print(f"任务路径点数: {len(waypoints)}")
    print(f"服务器: {host}:{port}")
    print(f"发送间隔: {delay} 秒")
    print("="*70 + "\n")
    
    success_count = 0
    
    for i, waypoint in enumerate(waypoints):
        print(f"[{i+1}/{len(waypoints)}] 发送路径点: {waypoint}")
        
        response = send_command('target', target=waypoint, host=host, port=port)
        
        if response['status'] == 'success':
            print(f"  ✅ {response['message']}")
            success_count += 1
        else:
            print(f"  ❌ {response['message']}")
        
        # 等待间隔（最后一个不需要等待）
        if i < len(waypoints) - 1:
            time.sleep(delay)
    
    print("\n" + "="*70)
    print(f"📊 任务完成统计")
    print("="*70)
    print(f"总路径点数: {len(waypoints)}")
    print(f"成功发送: {success_count}")
    print(f"发送失败: {len(waypoints) - success_count}")
    print("="*70 + "\n")


# 预定义任务示例
MISSIONS = {
    'square': [
        # 正方形巡航
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],  # 返回中心
    ],
    
    'zigzag': [
        # Z 字形巡航
        [1.0, 1.0, 0.5],
        [-1.0, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 1.5],
        [-1.0, -1.0, 1.5],
        [0.0, 0.0, 1.0],
    ],
    
    'spiral': [
        # 螺旋上升
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.7],
        [-0.5, 0.0, 0.9],
        [0.0, -0.5, 1.1],
        [0.5, 0.0, 1.3],
        [0.0, 0.5, 1.5],
        [0.0, 0.0, 1.0],
    ],
    
    'obstacle_test': [
        # 障碍物环境测试（避开预设障碍物）
        [0.0, 0.0, 0.5],    # 起点
        [1.0, 0.0, 0.5],    # 右侧（避开 [0.5, 0.5, 0.5] 障碍物）
        [1.0, 1.0, 1.0],    # 右上
        [0.0, 1.0, 1.0],    # 中上（避开 [-0.5, 0.5, 0.5] 障碍物）
        [-1.0, 1.0, 0.5],   # 左上
        [-1.0, 0.0, 0.5],   # 左侧（避开 [0.0, -0.5, 0.5] 障碍物）
        [0.0, 0.0, 1.0],    # 返回中心并上升
    ],
}


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SAC 连续导航批量任务执行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用任务:
  square        - 正方形巡航 (5 个路径点)
  zigzag        - Z 字形巡航 (7 个路径点)
  spiral        - 螺旋上升 (7 个路径点)
  obstacle_test - 障碍物环境测试 (7 个路径点)

示例:
  # 执行正方形巡航任务
  python mission_example.py --mission square
  
  # 执行障碍物测试任务，延长发送间隔到 3 秒
  python mission_example.py --mission obstacle_test --delay 3.0
  
  # 自定义任务（通过文件）
  python mission_example.py --file my_mission.json
        """
    )
    
    # 任务选择
    parser.add_argument('--mission', type=str, choices=list(MISSIONS.keys()),
                       help='选择预定义任务')
    parser.add_argument('--file', type=str,
                       help='从 JSON 文件加载自定义任务')
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='localhost',
                       help='服务器地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8888,
                       help='服务器端口 (默认: 8888)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='路径点发送间隔（秒，默认: 2.0）')
    
    args = parser.parse_args()
    
    # 加载任务
    if args.mission:
        waypoints = MISSIONS[args.mission]
        print(f"\n📋 加载预定义任务: {args.mission}")
    elif args.file:
        try:
            import json
            with open(args.file, 'r') as f:
                data = json.load(f)
                waypoints = data['waypoints']
            print(f"\n📋 从文件加载任务: {args.file}")
        except Exception as e:
            print(f"\n❌ 加载任务文件失败: {e}")
            return
    else:
        print("\n❌ 错误: 必须指定 --mission 或 --file")
        parser.print_help()
        return
    
    # 执行任务
    try:
        execute_mission(waypoints, host=args.host, port=args.port, delay=args.delay)
    except KeyboardInterrupt:
        print("\n\n⚠️ 任务被用户中断")
    except Exception as e:
        print(f"\n❌ 任务执行失败: {e}")


if __name__ == '__main__':
    main()
