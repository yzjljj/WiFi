import os
import pickle
import numpy as np

from data_gen import cyc_fea_train_datas_gen, cyc_fea_test_diffdatas_gen, cyc_fea_test_diffdaysdatas_gen
from data_gen_multi_fea import multi_fea_test_diffdaysdatas_gen

# 1. 加载模型
model_file = r"C:\Users\arise\Desktop\复现代码\KNN_Cyc_wifi\2025_12_25_wifi_IQ_STS_LTS\Results_Multi_Num_10_MultiClassifiers_CFOcomp_chaneq\final_models.pkl"
with open(model_file, 'rb') as f:
    models = pickle.load(f)

# 2. 准备新数据（必须和训练时特征维度一致！）
# 数据集目录节点
rxnode_name = "2025_12_23_wifi_IQ_STS_LTS"

# 参数设置
train_num = 1600        # 训练样本数
test_num = 2000         # 测试样本数
cyc_period = 10           # 周期数
chanBW = 'CBW20'        # 信道带宽
fea_len = (1 + cyc_period - 1) * (cyc_period - 1) // 2 #每个信号的特征维度

t_path = r'C:\Users\arise\Desktop\wifi_audio'  # 数据路径

# === 1. 数据生成 ===
XTestFrames, YTestLabel, file_len = multi_fea_test_diffdaysdatas_gen(
    test_num, rxnode_name, fea_len, t_path, cyc_period, chanBW)

samples_per_class = test_num * 6
XTestCell = []
YTestCell = []
for k in range(file_len):
    start_idx = k * samples_per_class
    end_idx = (k + 1) * samples_per_class
    XTestCell.append(XTestFrames[start_idx:end_idx, :])
    YTestCell.append(YTestLabel[start_idx:end_idx])


X_20M = XTestCell[0]  # 第一个文件夹的数据是20MHz
X_10M = XTestCell[1]  # 第二个文件夹的数据是10MHz
X_5M = XTestCell[2]   # 第三个文件夹的数据是5MHz

Y_20M = YTestCell[0]   #
Y_10M = YTestCell[1]   #
Y_5M = YTestCell[2]    #

# === 添加预测与评估代码 ===
from sklearn.metrics import accuracy_score

test_sets = {
    "20MHz": (X_20M, Y_20M),
    "10MHz": (X_10M, Y_10M),
    "5MHz":  (X_5M, Y_5M)
}

results = {}
classifier_names = ['KNN', 'SVM_RBF', 'RandomForest', 'LightGBM']

for name in classifier_names:
    model = models[name]
    accs = {}
    for bw, (X, y) in test_sets.items():
        if len(y) == 0:
            acc = 0.0
        else:
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred) * 100
        accs[bw] = acc
    accs['Average'] = np.mean([accs['20MHz'], accs['10MHz'], accs['5MHz']])
    results[name] = accs

# 打印表格
print("\n" + "="*70)
print(f"{'Model':<15} {'20MHz (%)':<12} {'10MHz (%)':<12} {'5MHz (%)':<12} {'Average (%)':<12}")
print("="*70)
for name in classifier_names:
    r = results[name]
    print(f"{name:<15} {r['20MHz']:<12.2f} {r['10MHz']:<12.2f} {r['5MHz']:<12.2f} {r['Average']:<12.2f}")
print("="*70)