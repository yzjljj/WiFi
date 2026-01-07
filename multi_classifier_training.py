import os
import numpy as np
from typing import List, Tuple
import time
import pickle

from data_gen import cyc_fea_test_diffdatas_gen, cyc_fea_train_datas_gen
from data_gen_multi_fea import multi_fea_train_datas_gen, multi_fea_test_diffdatas_gen
from advanced_classifiers import build_classifiers  # ← 替换为你实际的模块名


def fnn_cyc_sts_lts_train_multi(train_node_list: List[str], rxnode_name: str,
                          cyc_period: int, train_num: int, test_num: int,
                          fea_len: int, t_path: str, chanBW: str):
    """
    使用多种分类器（KNN/SVM/RF/GBDT）进行训练和评估
    """
    # === 1. 数据生成 ===
    XTrainFrames, YTrainLabel, XTestFrames_cos, YTestLabel_cos = multi_fea_train_datas_gen(
        train_node_list, train_num, fea_len, cyc_period, chanBW)

    XTestFrames, YTestLabel, file_len = multi_fea_test_diffdatas_gen(
        test_num, rxnode_name, fea_len, t_path, cyc_period, chanBW)

    samples_per_class = test_num * 6
    XTestCell = []
    YTestCell = []
    for k in range(file_len):
        start_idx = k * samples_per_class
        end_idx = (k + 1) * samples_per_class
        XTestCell.append(XTestFrames[start_idx:end_idx, :])
        YTestCell.append(YTestLabel[start_idx:end_idx])

    # === 2. 配置 ===
    train_times = 10
    classifiers_dict = build_classifiers()  # 获取所有分类器
    classifier_names = list(classifiers_dict.keys())

    # 创建保存路径
    ftemp = f"{rxnode_name}/Results_Multi_Num_{train_times}_MultiClassifiers_CFOcomp_chaneq"
    save_path = os.path.join(os.getcwd(), ftemp)
    os.makedirs(save_path, exist_ok=True)

    # === 3. 多次实验 ===
    all_results = []  # shape: (train_times, num_classifiers, 4)

    for i in range(train_times):
        print(f"\n=== Experiment {i+1}/{train_times} ===")
        exp_result = []

        for name, clf in classifiers_dict.items():
            print(f"  Training {name}...")

            # 深拷贝或重新实例化（防止状态污染）
            from sklearn.base import clone
            model = clone(clf)

            # 训练
            start_time = time.time()
            model.fit(XTrainFrames, YTrainLabel.ravel())
            train_time = time.time() - start_time

            # 评估
            acc_cos = evaluate_accuracy(model, XTestFrames_cos, YTestLabel_cos)
            acc_square = evaluate_accuracy(model, XTestCell[0], YTestCell[0])
            acc_triangle = evaluate_accuracy(model, XTestCell[1], YTestCell[1])
            acc_v = (acc_square + acc_triangle) / 2

            exp_result.append([acc_v, acc_cos, acc_square, acc_triangle])
            print(f"    {name}: acc_v={acc_v:.2f}%, acc_cos={acc_cos:.2f}%")

        all_results.append(exp_result)

    # === 4. 保存结果 ===
    all_results = np.array(all_results)  # shape: (10, 4, 4)
    results_name = os.path.join(save_path, 'multi_classifier_results.npy')
    np.save(results_name, all_results)

    # 打印平均性能
    mean_results = np.mean(all_results, axis=0)  # (4, 4)
    print("\n=== Average Results Over {} Runs ===".format(train_times))
    for idx, name in enumerate(classifier_names):
        avg_acc_v = mean_results[idx, 0]
        print(f"{name}: Avg acc_v = {avg_acc_v:.2f}%")

    # 可选：保存模型（只保存最后一次的）
    final_models = {}
    for name, clf in classifiers_dict.items():
        model = clone(clf)
        model.fit(XTrainFrames, YTrainLabel.ravel())
        final_models[name] = model

    with open(os.path.join(save_path, 'final_models.pkl'), 'wb') as f:
        pickle.dump(final_models, f)

    print(f"\nResults saved to: {save_path}")


def evaluate_accuracy(model, X_test, y_test) -> float:
    """辅助函数：计算准确率（%）"""
    if len(y_test) == 0:
        return 0.0
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test.ravel()) * 100
    return acc