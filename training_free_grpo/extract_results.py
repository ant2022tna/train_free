import numpy as np
from sklearn.metrics import (
    accuracy_score,                  # ACC
    normalized_mutual_info_score,    # NMI
    adjusted_rand_score,             # ARI
    f1_score,                        # F1
    recall_score,                    # Sensitivity/召回率
    precision_score,                 # Precision/精确率
    confusion_matrix,                # 用于计算Specificity
    matthews_corrcoef,               # MCC (Matthews相关系数)
    cohen_kappa_score,               # CK (Cohen's Kappa)
)
import json
import re
import random

def simple_process(jsonl_file_path):
    """简化版本"""
    
    def option_to_number(option):
        match = re.match(r'\(([A-Z]+)\)', option)
        if not match:
            return -1
        letters = match.group(1)
        num = 0
        for char in letters:
            num = num * 26 + (ord(char) - ord('A'))
        return num
    
    gt_numbers = []
    acc_array = []
    new_array = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                # groundtruth转换
                gt = data.get('groundtruth', '').strip()
                gt_num = option_to_number(gt)
                gt_numbers.append(gt_num)
                
                # reward处理
                reward = data.get('reward', 0)
                acc = 1 if reward == 1 else 0
                acc_array.append(acc)
                
                # 新数组生成
                if acc == 1:
                    new_array.append(gt_num)
                else:
                    choices = [i for i in range(9) if i != gt_num]
                    new_array.append(random.choice(choices) if choices else random.randint(0, 8))
    
    return gt_numbers, acc_array, new_array


def compute_specificity(y_true, y_pred):
    """计算特异性 (Specificity)"""
    cm = confusion_matrix(y_true, y_pred)
    # 对于二分类问题，cm的格式为：
    # [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # 对于多分类问题，可以计算每个类别的specificity，然后取平均
        specificities = []
        n_classes = cm.shape[0]
        for i in range(n_classes):
            # 计算真阴性：所有不是类别i的样本中被正确预测为不是类别i的样本数
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            # 计算假阳性：被错误预测为类别i的样本数
            fp = np.sum(cm[:, i]) - cm[i, i]
            if (tn + fp) > 0:
                specificities.append(tn / (tn + fp))
            else:
                specificities.append(0.0)
        specificity = np.mean(specificities)
    return specificity

def calculate_classification_metrics(y_true, y_pred, y_score=None):
    """
    计算分类性能指标
    
    参数:
    y_true: numpy数组，真实标签
    y_pred: numpy数组，预测标签
    
    返回:
    dict: 包含各种性能指标的字典
    """
    metrics = {}
    
    # 准确率 (Accuracy)
    metrics['ACC'] = accuracy_score(y_true, y_pred)

    metrics['F1'] = f1_score(y_true, y_pred, average='weighted')
    
    # 精确率 (Precision)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')

    # 特异性 (Specificity)
    metrics['Specificity'] = compute_specificity(y_true, y_pred)

    # 归一化互信息 (Normalized Mutual Information)
    #metrics['NMI'] = normalized_mutual_info_score(y_true, y_pred)
    
    # 调整兰德指数 (Adjusted Rand Index)
    #metrics['ARI'] = adjusted_rand_score(y_true, y_pred)
    
    # [None, 'micro', 'macro', 'weighted'] # , zero_division=0
    # F1分数 - 使用macro处理多分类问题，类别不平衡时，macro-F1更能反映小类的表现

    
    # 敏感度/召回率 (Sensitivity/Recall)
    #metrics['Sensitivity'] = recall_score(y_true, y_pred, average='weighted')
    

    # Matthews相关系数 (MCC)
    try:
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    except Exception as e:
        metrics['MCC'] = float('nan')
        print(f"无法计算MCC: {e}")
    
    # Cohen's Kappa (CK)
    try:
        metrics['CK'] = cohen_kappa_score(y_true, y_pred)
    except Exception as e:
        metrics['CK'] = float('nan')
        print(f"无法计算CK: {e}")

    
    return metrics

def print_metrics_table(metrics):
    """以表格形式打印指标"""
    print("\n" + "=" * 50)
    print("{:<15} {:<10}".format("指标", "值"))
    print("-" * 50)
    for metric_name, value in metrics.items():
        print("{:<15} {:<10.4f}".format(metric_name, value))
    print("=" * 50)


import scipy.io as sio
import pandas as pd

# 主程序
if __name__ == "__main__":
    # 加载NEJMQA_LLMs.mat数据
    gt, acc, new = simple_process("/home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/eval/medqa_gpt4o.jsonl")

    y_true=np.array(gt)    
    y_pred=np.array(new)
    
    metrics = calculate_classification_metrics(y_true, y_pred)
    print_metrics_table(metrics)