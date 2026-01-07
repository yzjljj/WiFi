"""
跨天射频指纹识别训练脚本
使用域适应技术提高时间泛化能力
"""

import os
import numpy as np
from typing import List, Tuple, Dict
import scipy.io as sio
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from domain_adaptation import (
    extract_domain_invariant_features,
    DomainAdapter,
    RFDataAugmentor,
    build_robust_classifiers,
    create_ensemble_classifier
)


def scan_dir(path: str, max_files: int = 2000) -> List[str]:
    """
    扫描目录下的文件，限制最大数量
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):  # 排序确保一致性
            if file.endswith('. mat'):
                file_list.append(os.path.join(root, file))
                if len(file_list) >= max_files:
                    return file_list
    return file_list


def load_signal(filepath: str) -> np.ndarray:
    """
    加载信号文件
    """
    try:
        data = sio.loadmat(filepath)
        sig = data['sig_data']. flatten()
        return sig
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def generate_domain_invariant_dataset(
    node_paths: List[str],
    num_samples: int,
    cyc_period: int,
    num_samples_per_cyc: int,
    augment:  bool = True,
    num_augments: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成域不变特征数据集
    
    Args:
        node_paths: 每个设备的数据路径列表
        num_samples: 每个设备的样本数
        cyc_period: 周期数 (10 for STS)
        num_samples_per_cyc: 每周期采样点数 (16)
        augment:  是否进行数据增强
        num_augments: 每个样本增强次数
    """
    features_list = []
    labels_list = []
    
    augmentor = RFDataAugmentor() if augment else None
    
    for device_idx, node_path in enumerate(node_paths):
        print(f"Processing device {device_idx + 1}:  {os.path.basename(node_path)}")
        
        file_list = scan_dir(node_path, max_files=num_samples)
        
        for filepath in file_list: 
            sig = load_signal(filepath)
            if sig is None:
                continue
            
            # 提取域不变特征
            try:
                features = extract_domain_invariant_features(sig, cyc_period, num_samples_per_cyc)
                features_list.append(features)
                labels_list.append(device_idx + 1)
                
                # 数据增强
                if augment and augmentor is not None:
                    aug_signals = augmentor. augment(sig, num_augments=num_augments)
                    for aug_sig in aug_signals[1:]:  # 跳过原始信号
                        aug_features = extract_domain_invariant_features(aug_sig, cyc_period, num_samples_per_cyc)
                        features_list.append(aug_features)
                        labels_list. append(device_idx + 1)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
    
    return np.array(features_list), np.array(labels_list)


def train_with_domain_adaptation(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np. ndarray,
    target_labels: np.ndarray,
    adaptation_method: str = 'coral'
) -> Dict: 
    """
    使用域适应进行训练和评估
    
    Args:
        source_features: 源域（训练）特征
        source_labels:  源域标签
        target_features: 目标域（测试）特征
        target_labels: 目标域标签
        adaptation_method:  域适应方法
    """
    results = {}
    
    # 标准化
    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_features)
    target_scaled = scaler.transform(target_features)
    
    # 域适应
    adapter = DomainAdapter(method=adaptation_method)
    adapter.fit(source_scaled)
    target_adapted = adapter.transform(target_scaled)
    
    # 构建分类器
    classifiers = build_robust_classifiers()
    
    print(f"\n=== 使用 {adaptation_method. upper()} 域适应 ===")
    print("-" * 50)
    
    for name, clf in classifiers.items():
        model = clone(clf)
        
        # 训练
        start_time = time. time()
        model.fit(source_scaled, source_labels)
        train_time = time.time() - start_time
        
        # 不使用域适应的准确率
        y_pred_no_adapt = model. predict(target_scaled)
        acc_no_adapt = accuracy_score(target_labels, y_pred_no_adapt) * 100
        
        # 使用域适应的准确率
        y_pred_adapted = model.predict(target_adapted)
        acc_adapted = accuracy_score(target_labels, y_pred_adapted) * 100
        
        improvement = acc_adapted - acc_no_adapt
        
        print(f"{name: 15s}: 无适应={acc_no_adapt:.2f}%, 有适应={acc_adapted:.2f}%, 提升={improvement: +.2f}%")
        
        results[name] = {
            'accuracy_no_adapt': acc_no_adapt,
            'accuracy_adapted': acc_adapted,
            'improvement': improvement,
            'train_time':  train_time,
            'predictions': y_pred_adapted,
            'model': model
        }
    
    return results


def cross_day_training_pipeline(
    train_path: str,
    test_paths: List[str],
    train_rxnode_name: str,
    device_list: List[str],
    train_num:  int = 1500,
    test_num: int = 500,
    cyc_period: int = 10,
    num_samples_per_cyc:  int = 16,
    use_augmentation: bool = True,
    adaptation_method: str = 'coral'
):
    """
    完整的跨天训练流水线
    
    Args:
        train_path: 训练数据根路径
        test_paths: 测试数据路径列表（不同天/不同参数）
        train_rxnode_name: 训练数据子目录名（如 '2025_12_23_wifi_IQ'）
        device_list: 设备名列表
        train_num: 训练样本数（每设备）
        test_num: 测试样本数（每设备）
        cyc_period: STS周期数
        num_samples_per_cyc: 每周期采样点
        use_augmentation: 是否使用数据增强
        adaptation_method: 域适应方法
    """
    print("=" * 60)
    print("跨天射频指纹识别 - 域适应训练")
    print("=" * 60)
    
    # 构建训练数据路径
    train_node_paths = [
        os.path. join(train_path, 'Config_wifi_20M', train_rxnode_name, device)
        for device in device_list
    ]
    
    # 生成训练数据
    print("\n[1/4] 生成训练数据（域不变特征 + 数据增强）...")
    train_features, train_labels = generate_domain_invariant_dataset(
        train_node_paths, train_num, cyc_period, num_samples_per_cyc,
        augment=use_augmentation, num_augments=3
    )
    print(f"训练数据形状:  {train_features. shape}")
    
    # 处理不同测试集
    all_results = {}
    
    for test_idx, test_subpath in enumerate(test_paths):
        test_name = os.path. basename(test_subpath)
        print(f"\n[{test_idx+2}/4] 处理测试集:  {test_name}")
        
        # 构建测试数据路径
        test_node_paths = [
            os. path.join(test_subpath, device)
            for device in device_list
        ]
        
        # 检查路径是否存在
        valid_paths = [p for p in test_node_paths if os.path.exists(p)]
        if len(valid_paths) != len(test_node_paths):
            print(f"警告: 某些设备路径不存在，跳过此测试集")
            continue
        
        # 生成测试数据
        test_features, test_labels = generate_domain_invariant_dataset(
            test_node_paths, test_num, cyc_period, num_samples_per_cyc,
            augment=False
        )
        print(f"测试数据形状:  {test_features. shape}")
        
        # 训练和评估
        results = train_with_domain_adaptation(
            train_features, train_labels,
            test_features, test_labels,
            adaptation_method=adaptation_method
        )
        
        all_results[test_name] = results
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    
    summary_table = []
    classifier_names = list(build_robust_classifiers().keys())
    
    for clf_name in classifier_names:
        row = {'Model': clf_name}
        for test_name, results in all_results.items():
            if clf_name in results: 
                row[test_name] = results[clf_name]['accuracy_adapted']
        summary_table.append(row)
    
    # 打印表格
    print("\n使用域适应后的识别准确率 (%):")
    print("-" * 80)
    header = "Model". ljust(15) + "". join([name[: 12]. center(15) for name in all_results.keys()])
    print(header)
    print("-" * 80)
    
    for row in summary_table:
        line = row['Model'].ljust(15)
        for test_name in all_results. keys():
            if test_name in row:
                line += f"{row[test_name]:.1f}%".center(15)
            else:
                line += "N/A".center(15)
        print(line)
    
    return all_results


# ============== 主程序入口 ==============

if __name__ == "__main__": 
    # 配置参数
    BASE_PATH = r"C:\Users\arise\Desktop\lora_audio"
    
    # 训练数据配置
    TRAIN_RXNODE = "2025_12_23_wifi_IQ"  # 训练日期
    
    # 设备列表
    DEVICE_LIST = [
        'hackrf_5c63',
        'hackrf_70cf', 
        'hackrf_5453',
        'hackrf_7353',
        'hackrf_8783',
        'hackrf_9583'
    ]
    
    # 测试集路径（不同带宽或不同天）
    TEST_PATHS = [
        os. path.join(BASE_PATH, 'Config_wifi_20M', TRAIN_RXNODE),  # 同天同参数
        os.path.join(BASE_PATH, 'Config_wifi_10M', TRAIN_RXNODE),  # 同天不同带宽
        os.path.join(BASE_PATH, 'Config_wifi_5M', TRAIN_RXNODE),   # 同天不同带宽
    ]
    
    # 如果有跨天数据，添加到这里
    # TEST_PATHS. append(os.path. join(BASE_PATH, 'Config_wifi_20M', '2025_12_24_wifi_IQ'))
    
    # 运行训练
    results = cross_day_training_pipeline(
        train_path=BASE_PATH,
        test_paths=TEST_PATHS,
        train_rxnode_name=TRAIN_RXNODE,
        device_list=DEVICE_LIST,
        train_num=1500,       # 每设备训练样本数
        test_num=500,         # 每设备测试样本数
        cyc_period=10,        # STS周期数
        num_samples_per_cyc=16,
        use_augmentation=True,
        adaptation_method='coral'  # 可选:  'coral', 'mmd', 'subspace'
    )