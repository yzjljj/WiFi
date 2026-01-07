
import os
import numpy as np
from typing import List, Tuple
import scipy.io as sio
from scipy.linalg import bandwidth

from cfo_est import estimate_cfo_multi_sts
from chann_eq import equalize_full_preamble_adaptive
from data_gen import compensate_cfo
from enhanced_data_gen import enhanced_cyc_similarity_features

import re


def extract_bandwidth(path_name: str) -> str:
    """
    从 path_name 中提取 WiFi 带宽并返回 'XXMHz' 格式的字符串。
    例如：'Config_wifi_20M' → '20MHz'

    Args:
        path_name (str): 包含如 'Config_wifi_20M' 的字符串

    Returns:
        str: 带宽字符串，如 '20MHz'、'10MHz'、'5MHz'

    Raises:
        ValueError: 如果未找到匹配的带宽模式
    """
    match = re.search(r'Config_wifi_(\d+)M', path_name)
    if match:
        return match.group(1) + 'M'
    else:
        raise ValueError(f"无法从 path_name 中提取带宽信息: {path_name}")

def parse_chirp_config(path_str):
    """
    根据路径字符串中的带宽标识（CBW20/CBW10/CBW5）解析 chirp 采样点数。

    参数:
        path_str (str): 包含带宽信息的路径或配置字符串

    返回:
        int: numSamples，当前固定为 16

    异常:
        ValueError: 如果未识别到支持的带宽标识
    """
    # 确保输入是字符串（兼容 str 类型）
    if not isinstance(path_str, str):
        raise TypeError("path_str must be a string")

    # 检查是否包含支持的带宽标识（不区分大小写更健壮，但原逻辑区分）
    if "CBW20" in path_str or "CBW10" in path_str or "CBW5" in path_str:
        num_samples = 16
    else:
        raise ValueError("未识别的带宽（需包含 CBW20/CBW10/CBW5）")

    return num_samples

def cyc_similarity_features(sig: np.ndarray, cyc_period:int, numSamples:int) -> np.ndarray:
    """
    计算信号的周期性特征
    参数:
        sig: 输入信号
    返回:
        coffe_list: 特征向量
    """
    cyc_len = numSamples

    # 信号归一化
    sig_c = sig / np.sqrt(np.mean(np.abs(sig) ** 2))  # 相当于MATLAB的rms

    # 初始化特征列表
    coffe_list = []

    # 计算段训练符号的重复性
    for k in range(cyc_period):
        index1 = slice(k * cyc_len, (k + 1) * cyc_len)
        sig_t1 = sig_c[index1]

        for l in range(k + 1, cyc_period):
            index2 = slice(l * cyc_len, (l + 1) * cyc_len)
            sig_t2 = sig_c[index2]

            # 计算特征
            coffe_real, coffe_imag, dealtf = cyc_fea_coffe_shape_sim(sig_t1, sig_t2)

            coffe_list.extend([coffe_real, coffe_imag])

    return np.array(coffe_list)


def cyc_fea_coffe_shape_sim(sig_t1: np.ndarray, sig_t2: np.ndarray) -> Tuple[float, float, float]:
    """
    计算两个信号段之间的特征
    参数:
        sig_t1: 第一个信号段
        sig_t2: 第二个信号段
    返回:
        coffe_real: 实部相关系数
        coffe_imag: 虚部相关系数
        dealtf: 相位差
    """
    # 计算相关系数（注意：Python中需要先转置再计算内积）
    coffe = np.dot(sig_t1.conj().T, sig_t2)

    # 归一化
    coffe = coffe / (np.linalg.norm(sig_t1) * np.linalg.norm(sig_t2))

    # 提取实部和虚部
    coffe_real = np.real(coffe)
    coffe_imag = np.imag(coffe)

    # 计算相位差
    dealtf = np.angle(coffe)

    return coffe_real, coffe_imag, dealtf



def multi_fea_train_datas_gen(nodel_list: List[str],
                           train_num: int, fea_len: int, cyc_period:int, chanBW: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成训练数据和测试数据
    参数:
        nodel_list: 节点列表
        train_num: 训练样本数
        fea_len: 特征维度
    返回:
        train_frame: 训练数据
        train_label: 训练标签
        test_frame: 测试数据
        test_label: 测试标签
    """
    # 计算实际特征长度
    numSamples = parse_chirp_config(chanBW)

    path_name = nodel_list[0]
    bandwidth = extract_bandwidth(path_name)

    fea_toal = 172

    # 初始化训练数据数组
    train_frame = np.zeros((train_num * 6, fea_toal))
    train_label = np.zeros((train_num * 6, 1))

    # 初始化测试数据列表
    test_frame = []
    test_label = []

    index = 0

    # 遍历6个接收节点
    for i in range(6):
        # 获取当前节点路径下的所有文件
        filelist = scan_dir(nodel_list[i])  # scan_dir是自定义函数，用于扫描指定路径下的文件
        len_sig = len(filelist)  # 当前节点下信号文件的数量
        train_len = 0

        # 遍历当前节点下的所有信号文件
        for j in range(len_sig):
            filename = filelist[j]  # 获取当前文件的路径

            # 加载信号文件
            try:
                # 尝试加载.mat文件
                data = sio.loadmat(filename)
                sig = data['sig_data']
            except:
                # 如果不是.mat文件，尝试其他格式
                sig = np.load(filename)

            # CFO估计与补偿
            fs = 20e6
            cfo_norm = estimate_cfo_multi_sts(sig, cyc_period, 16)
            cfo_hz = cfo_norm * fs if not np.isnan(cfo_norm) else 0.0
            sig_comp = compensate_cfo(sig, cfo_hz, fs)
            enhanced_fea = enhanced_cyc_similarity_features(sig_comp, cyc_period, numSamples)

            # sig_eq = equalize_full_preamble_adaptive(sig_comp.squeeze(), bandwidth)
            # enhanced_fea = enhanced_cyc_similarity_features(sig_eq.reshape(-1, 1) , cyc_period, numSamples)


            # 如果还没有达到训练样本的数量
            if train_len < train_num:
                index += 1  # 更新训练集索引
                train_len += 1  # 增加训练样本数量
                train_frame[index - 1, :] = enhanced_fea  # 将特征添加到训练集
                train_label[index - 1, 0] = i + 1  # 将当前节点的标签添加到训练标签（注意：Python索引从0开始）
            else:
                # 如果训练样本已经够了，则将当前样本作为测试集数据
                test_frame.append(enhanced_fea)  # 将特征添加到测试集
                test_label.append(i + 1)  # 将标签添加到测试集标签

    # 将测试数据转换为numpy数组
    test_frame = np.array(test_frame)
    test_label = np.array(test_label).reshape(-1, 1)

    return train_frame, train_label, test_frame, test_label


def scan_dir(path: str) -> List[str]:
    """
    扫描指定路径下的所有文件
    参数:
        path: 要扫描的路径
    返回:
        文件路径列表
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def multi_fea_test_diffdatas_gen(test_num: int,
                                 rxnode_name: str, fea_len: int, t_path: str, cyc_period:int, chanBW: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成不同天测试数据的函数
    参数:
        test_num: 测试样本数
        rxnode_name: 接收节点名称
        fea_len: 特征维度
        t_path: 数据路径
    返回:
        frame: 测试数据
        label: 测试标签
    """
    # 计算实际特征长度
    fea_toal = 172
    numSamples = parse_chirp_config(chanBW)

    # 构建测试数据路径列表
    wifiday_rx_list = [
        os.path.join(t_path, 'Config_wifi_10M', rxnode_name),
        os.path.join(t_path, 'Config_wifi_5M', rxnode_name)
    ]
    file_len = len(wifiday_rx_list)
    # 获取所有接收节点路径
    rx_tx_nodel_list = []
    for j in range(file_len):  # 遍历三种类型的数据
        # 获取当前路径下的所有文件夹
        for item in os.listdir(wifiday_rx_list[j]):
            dir_path = os.path.join(wifiday_rx_list[j], item)
            if os.path.isdir(dir_path):
                rx_tx_nodel_list.append(dir_path)

    # 初始化测试数据数组
    frame = np.zeros((test_num * 6 * file_len, fea_toal))
    label = np.zeros((test_num * 6 * file_len, 1))

    index = 0
    # 遍历所有接收节点
    for i in range(len(rx_tx_nodel_list)):
        # 获取当前节点下的所有文件
        filelist = scan_dir(rx_tx_nodel_list[i])
        temp_len = 0
        path_name = rx_tx_nodel_list[i]
        bandwidth = extract_bandwidth(path_name)
        # 遍历当前节点下的所有文件
        for j in range(len(filelist)):
            filename = filelist[j]

            # 加载信号文件
            try:
                # 尝试加载.mat文件
                data = sio.loadmat(filename)
                sig = data['sig_data']
            except:
                # 如果不是.mat文件，尝试其他格式
                sig = np.load(filename)

            # CFO估计与补偿
            fs = 20e6
            cfo_norm = estimate_cfo_multi_sts(sig, cyc_period, 16)
            cfo_hz = cfo_norm * fs if not np.isnan(cfo_norm) else 0.0
            sig_comp = compensate_cfo(sig, cfo_hz, fs)
            enhanced_fea = enhanced_cyc_similarity_features(sig_comp, cyc_period, numSamples)

            # sig_eq = equalize_full_preamble_adaptive(sig_comp.squeeze(), bandwidth)
            # enhanced_fea = enhanced_cyc_similarity_features(sig_eq.reshape(-1, 1), cyc_period, numSamples)
            # # 确定标签类型
            if (i + 1) % 6 == 0:
                type_label = 6
            else:
                type_label = (i + 1) % 6

            # 更新计数器和索引
            temp_len += 1
            index += 1

            # 存储特征和标签
            frame[index - 1, :] = enhanced_fea
            label[index - 1, 0] = type_label

            # 如果达到测试样本数量，跳出循环
            if temp_len > test_num - 1:
                break

    return frame, label, file_len

def multi_fea_test_diffdaysdatas_gen(test_num: int,
                                 rxnode_name: str, fea_len: int, t_path: str, cyc_period:int, chanBW: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成不同天测试数据的函数
    参数:
        test_num: 测试样本数
        rxnode_name: 接收节点名称
        fea_len: 特征维度
        t_path: 数据路径
    返回:
        frame: 测试数据
        label: 测试标签
    """
    # 计算实际特征长度
    fea_toal = 172
    numSamples = parse_chirp_config(chanBW)

    # 构建测试数据路径列表
    wifiday_rx_list = [
        os.path.join(t_path, 'Config_wifi_20M', rxnode_name),
        os.path.join(t_path, 'Config_wifi_10M', rxnode_name),
        os.path.join(t_path, 'Config_wifi_5M', rxnode_name)
    ]
    file_len = len(wifiday_rx_list)
    # 获取所有接收节点路径
    rx_tx_nodel_list = []
    for j in range(file_len):  # 遍历三种类型的数据
        # 获取当前路径下的所有文件夹
        for item in os.listdir(wifiday_rx_list[j]):
            dir_path = os.path.join(wifiday_rx_list[j], item)
            if os.path.isdir(dir_path):
                rx_tx_nodel_list.append(dir_path)

    # 初始化测试数据数组
    frame = np.zeros((test_num * 6 * file_len, fea_toal))
    label = np.zeros((test_num * 6 * file_len, 1))

    index = 0
    # 遍历所有接收节点
    for i in range(len(rx_tx_nodel_list)):
        # 获取当前节点下的所有文件
        filelist = scan_dir(rx_tx_nodel_list[i])
        temp_len = 0
        path_name = rx_tx_nodel_list[i]
        bandwidth = extract_bandwidth(path_name)
        # 遍历当前节点下的所有文件
        for j in range(len(filelist)):
            filename = filelist[j]

            # 加载信号文件
            try:
                # 尝试加载.mat文件
                data = sio.loadmat(filename)
                sig = data['sig_data']
            except:
                # 如果不是.mat文件，尝试其他格式
                sig = np.load(filename)

            # CFO估计与补偿
            fs = 20e6
            cfo_norm = estimate_cfo_multi_sts(sig, cyc_period, 16)
            cfo_hz = cfo_norm * fs if not np.isnan(cfo_norm) else 0.0
            sig_comp = compensate_cfo(sig, cfo_hz, fs)
            enhanced_fea = enhanced_cyc_similarity_features(sig_comp, cyc_period, numSamples)
            # enhanced_fea = enhanced_cyc_similarity_features(sig, cyc_period, numSamples)
            # sig_eq = equalize_full_preamble_adaptive(sig_comp.squeeze(), bandwidth)
            # enhanced_fea = enhanced_cyc_similarity_features(sig_eq.reshape(-1, 1), cyc_period, numSamples)

            # 确定标签类型
            if (i + 1) % 6 == 0:
                type_label = 6
            else:
                type_label = (i + 1) % 6

            # 更新计数器和索引
            temp_len += 1
            index += 1

            # 存储特征和标签
            frame[index - 1, :] = enhanced_fea
            label[index - 1, 0] = type_label

            # 如果达到测试样本数量，跳出循环
            if temp_len > test_num - 1:
                break

    return frame, label, file_len
