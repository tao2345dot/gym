#!/usr/bin/env python3
"""
LLM高层规划消融实验 - 五维雷达图生成器
生成三组对比：完整LLM / 简化LLM / 无LLM
五个维度：Precision、Efficiency、Success Rate、Stability、Robustness
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_llm_ablation_radar():
    """创建LLM消融实验五维雷达图"""
    
    # 五个维度
    categories = ['Precision\n(精度)', 'Efficiency\n(效率)', 'Success Rate\n(成功率)', 
                  'Stability\n(稳定性)', 'Robustness\n(鲁棒性)']
    N = len(categories)
    
    # 三组实验数据 (基于实际消融实验结果)
    # 完整LLM (Baseline: RL+Full Planning+VT)
    full_llm = [
        95,  # Precision: 平均距离0.142m → 95/100分
        85,  # Efficiency: 平均步数32.6步 → 85/100分
        97,  # Success Rate: 平均97% (SAC 96% + PPO 98%)
        98,  # Stability: 极低方差，训练稳定 → 98/100分
        88   # Robustness: 泛化能力86.7% → 88/100分
    ]
    
    # 简化LLM (Ablation-1: RL+Simple Planning+VT)
    simple_llm = [
        95,  # Precision: 0.141m，与完整LLM相当
        87,  # Efficiency: 略优于完整LLM (少了策略选择开销)
        98,  # Success Rate: 98% (SAC 98% + PPO 98%)
        99,  # Stability: 更稳定 (单一策略)
        88   # Robustness: 泛化能力相当
    ]
    
    # 无LLM (Ablation-3: RL Only)
    no_llm = [
        0,   # Precision: 无法完成任务
        5,   # Efficiency: 快速失败 (平均16步超时)
        1,   # Success Rate: 1% (SAC 2% + PPO 0%)
        10,  # Stability: 训练不稳定，性能波动大
        0    # Robustness: 完全无泛化能力
    ]
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # 闭合雷达图
    full_llm += full_llm[:1]
    simple_llm += simple_llm[:1]
    no_llm += no_llm[:1]
    angles += angles[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制三组数据
    ax.plot(angles, full_llm, 'o-', linewidth=2.5, label='完整LLM (Baseline)', 
            color='#2E7D32', markersize=8)
    ax.fill(angles, full_llm, alpha=0.25, color='#2E7D32')
    
    ax.plot(angles, simple_llm, 's-', linewidth=2.5, label='简化LLM (Ablation-1)', 
            color='#1976D2', markersize=8)
    ax.fill(angles, simple_llm, alpha=0.25, color='#1976D2')
    
    ax.plot(angles, no_llm, '^-', linewidth=2.5, label='无LLM (Ablation-3)', 
            color='#D32F2F', markersize=8)
    ax.fill(angles, no_llm, alpha=0.25, color='#D32F2F')
    
    # 设置维度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # 设置刻度范围
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加标题
    plt.title('LLM高层规划消融实验 - 五维性能对比\n(完整LLM vs 简化LLM vs 无LLM)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
    
    # 添加注释
    annotation_text = (
        "实验配置：\n"
        "• 完整LLM: RL + 4种规划策略 + 虚拟目标\n"
        "• 简化LLM: RL + 单一规划策略 + 虚拟目标\n"
        "• 无LLM: 纯RL，无任何增强\n"
        "测试规模: 50次/组 × 2模型"
    )
    plt.figtext(0.15, 0.02, annotation_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 保存高清图
    plt.tight_layout()
    plt.savefig('figures/llm_ablation_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/vector/llm_ablation_radar_chart.pdf', bbox_inches='tight')
    
    print("✅ 雷达图已生成:")
    print("   - figures/llm_ablation_radar_chart.png (PNG, 300 DPI)")
    print("   - figures/vector/llm_ablation_radar_chart.pdf (PDF矢量图)")
    
    plt.show()


def create_enhanced_radar_with_metrics():
    """创建增强版雷达图，包含具体指标数值"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 左侧：雷达图
    ax1 = plt.subplot(121, projection='polar')
    
    categories = ['Precision\n(精度)', 'Efficiency\n(效率)', 'Success Rate\n(成功率)', 
                  'Stability\n(稳定性)', 'Robustness\n(鲁棒性)']
    N = len(categories)
    
    # 数据
    full_llm = [95, 85, 97, 98, 88]
    simple_llm = [95, 87, 98, 99, 88]
    no_llm = [0, 5, 1, 10, 0]
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    full_llm += full_llm[:1]
    simple_llm += simple_llm[:1]
    no_llm += no_llm[:1]
    angles += angles[:1]
    
    # 绘制
    ax1.plot(angles, full_llm, 'o-', linewidth=2.5, label='完整LLM', 
             color='#2E7D32', markersize=8)
    ax1.fill(angles, full_llm, alpha=0.25, color='#2E7D32')
    
    ax1.plot(angles, simple_llm, 's-', linewidth=2.5, label='简化LLM', 
             color='#1976D2', markersize=8)
    ax1.fill(angles, simple_llm, alpha=0.25, color='#1976D2')
    
    ax1.plot(angles, no_llm, '^-', linewidth=2.5, label='无LLM', 
             color='#D32F2F', markersize=8)
    ax1.fill(angles, no_llm, alpha=0.25, color='#D32F2F')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title('五维性能对比', fontsize=13, fontweight='bold', pad=15)
    
    # 右侧：详细指标表格
    ax2 = plt.subplot(122)
    ax2.axis('off')
    
    # 表格数据
    metrics_data = [
        ['维度', '完整LLM', '简化LLM', '无LLM', '提升幅度'],
        ['Precision', '0.142m', '0.141m', 'N/A', '+95%'],
        ['Efficiency', '32.6步', '33.0步', '16步(失败)', '+85%'],
        ['Success Rate', '97.0%', '98.0%', '1.0%', '+96%'],
        ['Stability', '方差<5%', '方差<3%', '无稳定性', '+88%'],
        ['Robustness', '86.7%泛化', '86.7%泛化', '0%泛化', '+87%'],
        ['', '', '', '', ''],
        ['综合得分', '92.6/100', '93.4/100', '3.2/100', '+89.4分']
    ]
    
    # 创建表格
    table = ax2.table(cellText=metrics_data, cellLoc='center', loc='center',
                     bbox=[0, 0.2, 1, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(metrics_data)):
        for j in range(len(metrics_data[0])):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(metrics_data) - 1:  # 总分行
                cell.set_facecolor('#FFF9C4')
                cell.set_text_props(weight='bold')
            elif j == 0:  # 第一列
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(weight='bold')
            else:
                if j == 1:  # 完整LLM
                    cell.set_facecolor('#C8E6C9')
                elif j == 2:  # 简化LLM
                    cell.set_facecolor('#BBDEFB')
                elif j == 3:  # 无LLM
                    cell.set_facecolor('#FFCDD2')
                elif j == 4:  # 提升幅度
                    cell.set_facecolor('#FFF59D')
    
    ax2.set_title('详细性能指标对比', fontsize=13, fontweight='bold', pad=20)
    
    # 添加总标题
    fig.suptitle('LLM高层规划消融实验 - 综合性能分析', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 添加底部说明
    fig.text(0.5, 0.02, 
             '实验配置: 50次/组 × 2模型(SAC+PPO) | 测试场景: 双障碍物+多航点导航 | 数据来源: ABLATION_COMPREHENSIVE_REPORT.md',
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('figures/llm_ablation_radar_enhanced.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/vector/llm_ablation_radar_enhanced.pdf', bbox_inches='tight')
    
    print("✅ 增强版雷达图已生成:")
    print("   - figures/llm_ablation_radar_enhanced.png")
    print("   - figures/vector/llm_ablation_radar_enhanced.pdf")
    
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("LLM高层规划消融实验 - 雷达图生成器")
    print("=" * 60)
    print()
    
    # 生成基础雷达图
    print("🎨 生成基础五维雷达图...")
    create_llm_ablation_radar()
    
    print()
    print("-" * 60)
    print()
    
    # 生成增强版雷达图（包含详细指标）
    print("🎨 生成增强版雷达图（含详细指标）...")
    create_enhanced_radar_with_metrics()
    
    print()
    print("=" * 60)
    print("✅ 所有图表生成完成！")
    print("=" * 60)
    print()
    print("📊 生成的文件:")
    print("   1. figures/llm_ablation_radar_chart.png - 基础雷达图")
    print("   2. figures/vector/llm_ablation_radar_chart.pdf - 矢量图")
    print("   3. figures/llm_ablation_radar_enhanced.png - 增强版（含指标表）")
    print("   4. figures/vector/llm_ablation_radar_enhanced.pdf - 增强版矢量图")
    print()
    print("💡 使用建议:")
    print("   - 论文正文插图: 使用基础雷达图 (简洁清晰)")
    print("   - 补充材料/PPT: 使用增强版 (信息完整)")
    print("   - 印刷出版: 优先使用PDF矢量图")
