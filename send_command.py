"""
网络命令发送工具

用于向运行中的 SAC 连续导航系统发送目标点命令
"""

import socket
import json
import argparse


def send_command(command_type: str, target=None, host='localhost', port=8888):
    """
    发送命令到导航系统
    
    参数:
        command_type: 命令类型 ('target' 或 'home')
        target: 目标位置 [x, y, z] (仅 target 类型需要)
        host: 服务器地址
        port: 服务器端口
        
    返回:
        dict: 服务器响应
    """
    # 构建命令
    if command_type == 'target':
        if target is None:
            raise ValueError("target 命令需要提供目标位置")
        command = {
            'type': 'target',
            'target': target
        }
    elif command_type == 'home':
        command = {
            'type': 'home'
        }
    else:
        raise ValueError(f"未知命令类型: {command_type}")
    
    # 发送命令
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)  # 5秒超时
        sock.connect((host, port))
        
        # 发送 JSON 数据
        sock.send(json.dumps(command).encode('utf-8'))
        
        # 接收响应
        response_data = sock.recv(1024)
        response = json.loads(response_data.decode('utf-8'))
        
        sock.close()
        return response
        
    except socket.timeout:
        return {'status': 'error', 'message': '连接超时'}
    except ConnectionRefusedError:
        return {'status': 'error', 'message': '连接被拒绝，请确认导航系统已启动'}
    except Exception as e:
        return {'status': 'error', 'message': f'发送失败: {str(e)}'}


def main():
    """命令行工具主函数"""
    parser = argparse.ArgumentParser(
        description='向 SAC 连续导航系统发送命令',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 发送单个目标点
  python send_command.py --target 1.0 1.0 1.0
  
  # 发送多个目标点（连续导航）
  python send_command.py --target 1.0 1.0 1.0
  python send_command.py --target 1.0 -1.0 1.0
  python send_command.py --target -1.0 -1.0 0.5
  
  # 返回起点
  python send_command.py --home
  
  # 指定服务器地址和端口
  python send_command.py --target 1.0 1.0 1.0 --host 192.168.1.100 --port 8888
        """
    )
    
    # 命令类型（互斥）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--target', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                      help='发送目标点 (x, y, z)')
    group.add_argument('--home', action='store_true',
                      help='返回起点')
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='localhost',
                       help='服务器地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8888,
                       help='服务器端口 (默认: 8888)')
    
    args = parser.parse_args()
    
    # 发送命令
    print(f"\n📡 连接到 {args.host}:{args.port}...")
    
    if args.target:
        print(f"📍 发送目标: [{args.target[0]:.2f}, {args.target[1]:.2f}, {args.target[2]:.2f}]")
        response = send_command('target', target=args.target, host=args.host, port=args.port)
    else:
        print(f"🏠 请求返回起点")
        response = send_command('home', host=args.host, port=args.port)
    
    # 显示响应
    print("\n📥 服务器响应:")
    if response['status'] == 'success':
        print(f"✅ 成功: {response['message']}")
    else:
        print(f"❌ 失败: {response['message']}")
    print()


if __name__ == '__main__':
    main()
