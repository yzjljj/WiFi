import numpy as np


import pickle

# 打开并读取 .pkl 文件
path = r'C:\Users\arise\Desktop\复现代码\KNN_Cyc_wifi\CFO_Results_STS16\cfo_estimates_sts16.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
# 指定完整路径（注意：Windows 路径可以用正斜杠 / 或双反斜杠 \\）
file_path = r"C:\Users\arise\Desktop\复现代码\KNN_Cyc_wifi\2025_12_25_wifi_IQ\Results_Cycfea_Num_10_MultiClassifiers\multi_classifier_results.npy"

# 加载数据
results = np.load(file_path)

# 查看形状和内容
print("Shape:", results.shape)  # 应该是 (10, 4, 4)
print("Data type:", results.dtype)
print("Sample data:\n", results[0])  # 第一次实验的结果