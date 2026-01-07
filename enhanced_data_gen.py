import os
import numpy as np
from typing import List, Tuple
import scipy.io as sio
from scipy import signal
from scipy.fft import fft, fftshift


def enhanced_cyc_similarity_features(sig: np.ndarray, cyc_period:  int, numSamples: int) -> np.ndarray:
    """
    增强的周期性特征提取 - 融合多种特征
    """
    cyc_len = numSamples
    
    # 信号归一化
    sig_c = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    
    feature_list = []
    
    # 1. 原始Cyc-Similarity特征（实部和虚部）
    for k in range(cyc_period):
        index1 = slice(k * cyc_len, (k + 1) * cyc_len)
        sig_t1 = sig_c[index1]
        
        for l in range(k + 1, cyc_period):
            index2 = slice(l * cyc_len, (l + 1) * cyc_len)
            sig_t2 = sig_c[index2]
            
            # 复相关系数
            coffe = np.dot(sig_t1.conj().T, sig_t2)
            coffe = coffe / (np.linalg.norm(sig_t1) * np.linalg.norm(sig_t2) + 1e-10)

            # ✅ 转为标量！
            feature_list.append(float(np.real(coffe)))
            feature_list.append(float(np.imag(coffe)))
    
    # 2. 相位差特征 - 对带宽变化更鲁棒
    phase_features = extract_phase_features(sig_c, cyc_period, cyc_len)
    feature_list.extend(phase_features)
    
    # 3. 幅度统计特征 - 设备硬件非线性特征
    amp_features = extract_amplitude_features(sig_c, cyc_period, cyc_len)
    feature_list.extend(amp_features)
    
    # 4. 频域特征 - 捕获频率偏移和频谱特性
    freq_features = extract_frequency_features(sig_c)
    feature_list.extend(freq_features)
    
    return np.array(feature_list)


def extract_phase_features(sig_c: np.ndarray, cyc_period: int, cyc_len: int) -> List[float]:
    """
    提取相位差特征 - 相邻STS之间的相位变化
    """
    phase_features = []
    
    for k in range(cyc_period - 1):
        index1 = slice(k * cyc_len, (k + 1) * cyc_len)
        index2 = slice((k + 1) * cyc_len, (k + 2) * cyc_len)
        sig_t1 = sig_c[index1]
        sig_t2 = sig_c[index2]
        
        # 计算相位差
        phase_diff = np.angle(sig_t2) - np.angle(sig_t1)
        # 相位展开，处理2π跳变
        phase_diff = np.unwrap(phase_diff)
        
        # 统计特征：均值、方差、斜率
        phase_features.append(np.mean(phase_diff))
        phase_features.append(np.std(phase_diff))
        
        # 相位斜率（频率偏移指示）
        if len(phase_diff) >= 2:
            slope = np.polyfit(np.arange(len(phase_diff)), phase_diff, 1)[0]
        else:
            slope = 0.0
        phase_features.append(float(slope))  # ✅ 转 float
    
    return phase_features


def extract_amplitude_features(sig_c: np.ndarray, cyc_period: int, cyc_len: int) -> List[float]:
    """
    提取幅度统计特征 - 捕获功放非线性
    返回固定长度的 float 列表（每段 5 个特征 × cyc_period）
    """
    amp_features = []

    for k in range(cyc_period):
        index = slice(k * cyc_len, (k + 1) * cyc_len)
        sig_seg = sig_c[index]

        amp = np.abs(sig_seg)

        # 基础统计量（始终存在）
        mean_amp = float(np.mean(amp))  #均值
        std_amp = float(np.std(amp))    #标准差
        peak_to_peak = float(np.max(amp) - np.min(amp)) #峰峰值

        amp_features.extend([mean_amp, std_amp, peak_to_peak])

        # 高阶统计量：偏度和峰度（即使 std 很小也返回 0.0，保证维度固定）
        std_val = np.std(amp)
        if std_val < 1e-10:
            skewness = 0.0
            kurtosis = 0.0
        else:
            mean_val = np.mean(amp)
            skewness = np.mean((amp - mean_val) ** 3) / (std_val ** 3 + 1e-10)
            kurtosis = np.mean((amp - mean_val) ** 4) / (std_val ** 4 + 1e-10) - 3

        amp_features.append(float(skewness))
        amp_features.append(float(kurtosis))

    return amp_features


def extract_frequency_features(sig_c: np.ndarray) -> List[float]:
    """
    提取频域特征
    返回固定 5 个 float 特征
    """
    freq_features = []

    # FFT
    sig_fft = fftshift(fft(sig_c))
    mag_spectrum = np.abs(sig_fft)

    # 归一化频谱（避免除零）
    total_energy = float(np.sum(mag_spectrum))
    if total_energy < 1e-10:
        mag_spectrum_norm = mag_spectrum  # 或全零，但保持形状
    else:
        mag_spectrum_norm = mag_spectrum / total_energy

    # 1. 最大值
    max_mag = float(np.max(mag_spectrum_norm))
    freq_features.append(max_mag)

    # 2. 标准差
    std_mag = float(np.std(mag_spectrum_norm))
    freq_features.append(std_mag)

    # 3. 频谱质心
    n = len(mag_spectrum_norm)
    freqs = np.arange(n, dtype=np.float64)
    centroid = float(np.sum(freqs * mag_spectrum_norm) / (total_energy + 1e-10))
    freq_features.append(centroid)

    # 4. 频谱带宽
    bandwidth_sq = np.sum(((freqs - centroid) ** 2) * mag_spectrum_norm)
    bandwidth = float(np.sqrt(bandwidth_sq / (total_energy + 1e-10)))
    freq_features.append(bandwidth)

    # 5. 频谱平坦度（Spectral Flatness）
    # 防止 log(0)
    log_mag = np.log(mag_spectrum_norm + 1e-10)
    geometric_mean = float(np.exp(np.mean(log_mag)))
    arithmetic_mean = float(np.mean(mag_spectrum_norm))
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    freq_features.append(float(flatness))

    return freq_features