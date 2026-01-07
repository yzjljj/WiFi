import os

from cyc_knn_training import fnn_cyc_sts_lts_train
from multi_classifier_training import fnn_cyc_sts_lts_train_multi


def main():
    # 清空控制台
    os.system('cls' if os.name == 'nt' else 'clear')

    # 数据集目录节点
    rxnode_name = "2026_1_6_wireless_1_wifi_IQ"

    # 参数设置
    train_num = 1600        # 训练样本数
    test_num = 2000         # 测试样本数
    cyc_period = 10           # 周期数
    chanBW = 'CBW20'        # 信道带宽
    fea_len = (1 + cyc_period - 1) * (cyc_period - 1) // 2 #每个信号的特征维度

    t_path = r'C:\Users\arise\Desktop\wifi_audio'  # 数据路径



    dirroot = os.path.join(r'C:\Users\arise\Desktop\wifi_audio\Config_wifi_20M', rxnode_name)

    # 获取该路径下的所有文件夹
    train_node_list = []
    for item in os.listdir(dirroot):
        dir_path = os.path.join(dirroot, item)
        if os.path.isdir(dir_path):
            train_node_list.append(dir_path)


    fnn_cyc_sts_lts_train_multi(
        train_node_list,
        rxnode_name,
        cyc_period,
        train_num,
        test_num,
        fea_len,
        t_path,
        chanBW
    )


if __name__ == "__main__":
    main()