import re
import pandas as pd
import os
import json
from datetime import datetime

def parse_training_log(log_file_path):
    """
    解析训练日志文件并提取结果
    
    参数:
    log_file_path: 日志文件路径
    
    返回:
    包含解析结果的DataFrame
    """
    # 定义日志行的正则表达式模式
    window_result_pattern = re.compile(
        r"窗口 (\d+): 训练期 (\d{4}-\d{2}-\d{2}) 到 (\d{4}-\d{2}-\d{2}), "
        r"预测期 (\d{4}-\d{2}-\d{2}) 到 (\d{4}-\d{2}-\d{2}), "
        r"准确率 (\d+\.\d+), 标签1准确率 (\d+\.\d+), F1分数 (\d+\.\d+)"
    )
    
    summary_pattern = re.compile(
        r"平均准确率: (\d+\.\d+)"
    )
    
    label_1_summary_pattern = re.compile(
        r"平均标签1准确率: (\d+\.\d+)"
    )
    
    f1_summary_pattern = re.compile(
        r"平均F1分数: (\d+\.\d+)"
    )
    
    results = []
    summary_stats = {}
    
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 查找窗口结果
            window_match = window_result_pattern.search(line)
            if window_match:
                result = {
                    'window': int(window_match.group(1)),
                    'train_start': window_match.group(2),
                    'train_end': window_match.group(3),
                    'predict_start': window_match.group(4),
                    'predict_end': window_match.group(5),
                    'accuracy': float(window_match.group(6)),
                    'label_1_accuracy': float(window_match.group(7)),
                    'f1_score': float(window_match.group(8))
                }
                results.append(result)
                continue
            
            # 查找汇总统计
            if '平均准确率:' in line:
                summary_match = summary_pattern.search(line)
                if summary_match:
                    summary_stats['avg_accuracy'] = float(summary_match.group(1))
            
            if '平均标签1准确率:' in line:
                label_1_summary_match = label_1_summary_pattern.search(line)
                if label_1_summary_match:
                    summary_stats['avg_label_1_accuracy'] = float(label_1_summary_match.group(1))
            
            if '平均F1分数:' in line:
                f1_summary_match = f1_summary_pattern.search(line)
                if f1_summary_match:
                    summary_stats['avg_f1_score'] = float(f1_summary_match.group(1))
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame([summary_stats])
    
    return results_df, summary_df

def save_results_to_csv(results_df, summary_df, output_dir="results"):
    """
    将解析结果保存为CSV文件
    
    参数:
    results_df: 窗口结果DataFrame
    summary_df: 汇总统计DataFrame
    output_dir: 输出目录
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存窗口结果
    results_file = os.path.join(output_dir, f"window_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    
    # 保存汇总统计
    summary_file = os.path.join(output_dir, f"summary_stats_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"结果已保存到: {results_file}")
    print(f"汇总统计已保存到: {summary_file}")

def save_results_to_json(results_df, summary_df, output_dir="results"):
    """
    将解析结果保存为JSON文件
    
    参数:
    results_df: 窗口结果DataFrame
    summary_df: 汇总统计DataFrame
    output_dir: 输出目录
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 转换DataFrame为字典
    results_dict = {
        "window_results": results_df.to_dict('records'),
        "summary_stats": summary_df.to_dict('records')[0] if not summary_df.empty else {}
    }
    
    # 保存为JSON文件
    json_file = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"JSON结果已保存到: {json_file}")

def main():
    """
    主函数：解析日志并保存结果
    """
    # 日志文件路径
    log_file_path = os.path.join("logs", "training.log")
    
    # 检查日志文件是否存在
    if not os.path.exists(log_file_path):
        print(f"日志文件不存在: {log_file_path}")
        return
    
    # 解析日志文件
    print("正在解析日志文件...")
    results_df, summary_df = parse_training_log(log_file_path)
    
    if results_df.empty:
        print("未找到窗口结果数据")
        return
    
    # 显示解析结果
    print("\n=== 窗口结果 ===")
    print(results_df.to_string(index=False))
    
    print("\n=== 汇总统计 ===")
    print(summary_df.to_string(index=False))
    
    # 保存结果
    save_results_to_csv(results_df, summary_df)
    save_results_to_json(results_df, summary_df)

if __name__ == "__main__":
    main()