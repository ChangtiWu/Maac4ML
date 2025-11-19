#!/usr/bin/env python3
"""
实验结果绘图工具
从 jsonl 日志文件读取数据并进行可视化
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict


def load_experiment_data(log_file):
    """从 jsonl 文件加载实验数据"""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件不存在: {log_file}")

    # 使用 defaultdict 按实验ID组织数据
    experiments = defaultdict(lambda: {
        'name': '',
        'first_activation': '',
        'precision_decimal': None,
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_acc': [],
        'epoch_time': []
    })

    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                exp_id = data['experiment_id']
                exp = experiments[exp_id]

                # 保存实验元数据
                exp['name'] = data['experiment_name']
                exp['first_activation'] = data['first_activation']
                exp['precision_decimal'] = data['precision_decimal']

                # 保存epoch数据
                exp['epochs'].append(data['epoch'])
                exp['train_loss'].append(data['train_loss'])
                exp['train_acc'].append(data['train_acc'])
                exp['test_loss'].append(data['test_loss'])
                exp['test_acc'].append(data['test_acc'])
                exp['best_acc'].append(data['best_acc'])
                exp['epoch_time'].append(data['epoch_time'])

    return dict(experiments)


def plot_experiments(experiments, output_prefix='experiment_results'):
    """绘制实验对比图 - 分别保存三张独立的图"""
    # 准备数据
    exp_list = sorted(experiments.items(), key=lambda x: x[0])

    # 设置颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'd', 'v', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    # 提取实验名称（简化版）
    def simplify_name(name):
        if '实验1' in name:
            return 'Original precision + ReLU'
        elif '实验2' in name:
            return '3 decimal places + Quadratic'
        elif '实验3' in name:
            return '6 decimal places + ReLU (Ours)'
        return name

    # 获取输出目录和基础名称
    output_dir = os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.'
    base_name = os.path.basename(output_prefix)

    # 1. 训练损失对比
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, (_, exp_data) in enumerate(exp_list):
        ax.plot(exp_data['epochs'], exp_data['train_loss'],
                label=simplify_name(exp_data['name']),
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                markersize=6, linewidth=2.5, markevery=2)
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=18, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    train_loss_file = os.path.join(output_dir, f'{base_name}_train_loss.pdf')
    plt.savefig(train_loss_file, dpi=300, bbox_inches='tight')
    print(f"训练损失图已保存到: {train_loss_file}")
    plt.close()

    # 2. 训练准确率对比
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, (_, exp_data) in enumerate(exp_list):
        ax.plot(exp_data['epochs'], exp_data['train_acc'],
                label=simplify_name(exp_data['name']),
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                markersize=6, linewidth=2.5, markevery=2)
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Training Accuracy (%)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    train_acc_file = os.path.join(output_dir, f'{base_name}_train_acc.pdf')
    plt.savefig(train_acc_file, dpi=300, bbox_inches='tight')
    print(f"训练准确率图已保存到: {train_acc_file}")
    plt.close()

    # 3. 测试准确率对比 (最重要的指标)
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, (_, exp_data) in enumerate(exp_list):
        ax.plot(exp_data['epochs'], exp_data['test_acc'],
                label=simplify_name(exp_data['name']),
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                markersize=6, linewidth=2.5, markevery=2)
    ax.set_xlabel('Epoch', fontsize=18, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    test_acc_file = os.path.join(output_dir, f'{base_name}_test_acc.pdf')
    plt.savefig(test_acc_file, dpi=300, bbox_inches='tight')
    print(f"测试准确率图已保存到: {test_acc_file}")
    plt.close()


def print_summary(experiments):
    """打印实验总结"""
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)

    for _, exp_data in sorted(experiments.items()):
        print(f"\n{exp_data['name']}")
        print(f"  激活函数: {exp_data['first_activation']}")
        print(f"  精度控制: {exp_data['precision_decimal'] if exp_data['precision_decimal'] else '无限制'}")
        if exp_data['best_acc']:
            print(f"  最佳准确率: {max(exp_data['best_acc']):.2f}%")
            print(f"  最终准确率: {exp_data['test_acc'][-1]:.2f}%")
            print(f"  已完成轮数: {len(exp_data['epochs'])}")
            avg_time = sum(exp_data['epoch_time']) / len(exp_data['epoch_time'])
            print(f"  平均训练时间: {avg_time:.2f}秒/轮")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='实验结果可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python plot_experiments.py -l ../cnn/log/resnet_experiment_20241109_120000.jsonl
  python plot_experiments.py -l ../cnn/log/resnet_experiment_20241109_120000.jsonl -o results
        """
    )

    parser.add_argument('-l', '--log-file', required=True,
                       help='JSONL日志文件路径')
    parser.add_argument('-o', '--output', default='experiment_results',
                       help='输出图片文件前缀 (默认: experiment_results，将生成3个文件)')

    args = parser.parse_args()

    try:
        experiments = load_experiment_data(args.log_file)

        if not experiments:
            print("警告: 日志文件为空或没有有效数据")
            return

        print_summary(experiments)
        plot_experiments(experiments, args.output)
        print("\n所有图表已生成完成！")

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"处理数据时出错: {e}")
        raise


if __name__ == '__main__':
    main()
