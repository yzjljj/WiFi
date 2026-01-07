import os
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import KNeighborsClassifier
import time
import pickle

from data_gen import cyc_fea_test_diffdatas_gen, cyc_fea_train_datas_gen


def fnn_cyc_sts_lts_train(train_node_list: List[str], rxnode_name: str,
                            cyc_period: int, train_num: int, test_num: int,
                          fea_len: int, t_path: str, chanBW: str):
    """
    自定义训练函数（实现KNN分类器的训练和评估）
    """
    # 生成训练数据
    XTrainFrames, YTrainLabel, XTestFrames_cos, YTestLabel_cos = cyc_fea_train_datas_gen(
        train_node_list, train_num, fea_len, cyc_period, chanBW)

    # 生成所有测试数据
    XTestFrames, YTestLabel, file_len = cyc_fea_test_diffdatas_gen(
        test_num, rxnode_name, fea_len, t_path, cyc_period, chanBW)

    samples_per_class = test_num * 6  # 每类 6 * test_num 个样本


    # 预分配列表（Python 中用 list 代替 cell）
    XTestCell = [None] * file_len
    YTestCell = [None] * file_len

    # 动态分块
    for k in range(file_len):
        start_idx = k * samples_per_class  # Python 索引从 0 开始
        end_idx = (k + 1) * samples_per_class

        XTestCell[k] = XTestFrames[start_idx:end_idx, :]
        YTestCell[k] = YTestLabel[start_idx:end_idx]

    # KNN训练参数
    train_times = 10
    n_neighbors = np.arange(1, 1 + 40 * 6, 6)

    # 创建保存路径
    ftemp = f"{rxnode_name}/Results_CycFea_Num_{train_times}_KNN"
    save_path = os.path.join(os.getcwd(), ftemp)

    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # 存储结果
    results = []

    # 进行多次独立训练
    for i in range(train_times):
        # KNN训练
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=n_neighbors[i], weights='distance', metric='manhattan')
        knn.fit(XTrainFrames, YTrainLabel)
        run_times = time.time() - start_time

        # 保存模型
        model_name = os.path.join(save_path, f'model_{n_neighbors[i]}')
        with open(model_name, 'wb') as f:
            pickle.dump(knn, f)

        # 评估模型
        loss_v = 0
        acc_cos = knn_forest_classification_days(knn, XTestFrames_cos, YTestLabel_cos)[0]
        acc_square = knn_forest_classification_days(knn, XTestCell[0], YTestCell[0])[0]
        acc_triangle = knn_forest_classification_days(knn, XTestCell[1], YTestCell[1])[0]

        acc_v = (acc_square + acc_triangle) / 2
        each_result = [acc_v, acc_cos, acc_square, acc_triangle]
        results.append(each_result)

    # 保存结果
    results = np.array(results)
    results_name = os.path.join(save_path, 'total_result')
    np.save(results_name, results)
    print(results)


def knn_forest_classification_days(model, X_test, y_test) -> Tuple[float, int, np.ndarray, np.ndarray]:
    """
    KNN分类器评估函数
    参数:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
    返回:
        accuracy: 准确率（百分比）
        errorCount: 错误预测数量
        errorTrueLabels: 错误预测的真实标签
        errorPredLabels: 错误预测的预测标签
    """
    y_test = y_test.ravel()
    # 预测标签
    y_pred = model.predict(X_test)

    # 计算准确率
    correct_indices = (y_pred == y_test)
    accuracy = np.sum(correct_indices) / len(y_test) * 100

    # 找到预测错误的样本索引
    error_indices = np.where(~correct_indices)[0]
    error_count = len(error_indices)

    # 提取错误样本的真实标签和预测标签
    error_true_labels = y_test[error_indices]
    error_pred_labels = y_pred[error_indices]

    return accuracy, error_count, error_true_labels, error_pred_labels