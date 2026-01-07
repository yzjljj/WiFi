"""
域适应射频指纹识别模块
用于解决跨天/跨参数识别准确率低的问题
"""

import os
import numpy as np
from typing import List, Tuple, Optional
import scipy.io as sio
from scipy import signal
from scipy.fft import fft, fftshift
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ============== 1. 域不变特征提取 ==============

def extract_domain_invariant_features(sig:  np.ndarray, cyc_period: int, numSamples: int) -> np.ndarray:
    """
    提取域不变特征 - 对信道变化和时间漂移更鲁棒
    
    核心思想：
    1. 使用差分特征消除信道影响
    2. 使用归一化特征消除幅度漂移
    3. 使用相对特征而非绝对特征
    """
    cyc_len = numSamples
    sig = sig.flatten()
    
    # 信号归一化 - 消除整体幅度差异
    sig_c = sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-10)
    
    feature_list = []
    
    # 1. 差分相关特征 - 对信道变化更鲁棒
    diff_corr_features = extract_differential_correlation(sig_c, cyc_period, cyc_len)
    feature_list. extend(diff_corr_features)
    
    # 2. 相对相位特征 - 消除CFO漂移
    rel_phase_features = extract_relative_phase_features(sig_c, cyc_period, cyc_len)
    feature_list.extend(rel_phase_features)
    
    # 3. 归一化幅度特征 - 消除增益变化
    norm_amp_features = extract_normalized_amplitude_features(sig_c, cyc_period, cyc_len)
    feature_list. extend(norm_amp_features)
    
    # 4. 高阶累积量特征 - 对高斯噪声不敏感
    hoc_features = extract_higher_order_cumulants(sig_c)
    feature_list.extend(hoc_features)
    
    # 5. I/Q不平衡特征 - 硬件固有特性
    iq_features = extract_iq_imbalance_features(sig_c, cyc_period, cyc_len)
    feature_list. extend(iq_features)
    
    # 6. 非线性特征 - 功放非线性特性
    nonlinear_features = extract_nonlinearity_features(sig_c, cyc_period, cyc_len)
    feature_list.extend(nonlinear_features)
    
    return np.array(feature_list, dtype=np.float64)


def extract_differential_correlation(sig_c: np.ndarray, cyc_period: int, cyc_len: int) -> List[float]:
    """
    差分相关特征 - 计算相邻STS段之间的差分后再求相关
    这样可以消除信道的乘性影响
    """
    features = []
    
    for k in range(cyc_period - 1):
        index1 = slice(k * cyc_len, (k + 1) * cyc_len)
        index2 = slice((k + 1) * cyc_len, (k + 2) * cyc_len)
        
        sig_t1 = sig_c[index1]
        sig_t2 = sig_c[index2]
        
        # 差分操作 - 消除信道影响
        if len(sig_t1) > 1:
            diff1 = np.diff(sig_t1)
            diff2 = np.diff(sig_t2)
            
            # 差分信号的相关
            norm1 = np.linalg.norm(diff1) + 1e-10
            norm2 = np.linalg.norm(diff2) + 1e-10
            
            corr = np.dot(diff1.conj(), diff2) / (norm1 * norm2)
            features.append(float(np.abs(corr)))
            features.append(float(np.angle(corr)))
        else:
            features.extend([0.0, 0.0])
    
    return features


def extract_relative_phase_features(sig_c: np. ndarray, cyc_period: int, cyc_len:  int) -> List[float]:
    """
    相对相位特征 - 使用相位差的差分（二阶差分）
    消除载波频率偏移(CFO)的影响
    """
    features = []
    
    for k in range(cyc_period - 1):
        index1 = slice(k * cyc_len, (k + 1) * cyc_len)
        index2 = slice((k + 1) * cyc_len, (k + 2) * cyc_len)
        
        sig_t1 = sig_c[index1]
        sig_t2 = sig_c[index2]
        
        # 瞬时相位
        phase1 = np.angle(sig_t1)
        phase2 = np.angle(sig_t2)
        
        # 相位展开
        phase1_unwrap = np.unwrap(phase1)
        phase2_unwrap = np.unwrap(phase2)
        
        # 相位差的二阶差分 - 消除CFO
        if len(phase1_unwrap) > 2:
            phase_diff1 = np.diff(phase1_unwrap, n=2)
            phase_diff2 = np.diff(phase2_unwrap, n=2)
            
            # 统计特征
            features.append(float(np.mean(phase_diff1)))
            features.append(float(np.std(phase_diff1)))
            features. append(float(np.mean(phase_diff2 - phase_diff1)))
        else:
            features.extend([0.0, 0.0, 0.0])
    
    return features


def extract_normalized_amplitude_features(sig_c:  np.ndarray, cyc_period:  int, cyc_len: int) -> List[float]: 
    """
    归一化幅度特征 - 使用相对幅度变化
    """
    features = []
    
    # 计算每段的幅度统计
    amp_stats = []
    for k in range(cyc_period):
        index = slice(k * cyc_len, (k + 1) * cyc_len)
        seg = sig_c[index]
        amp = np.abs(seg)
        amp_stats.append({
            'mean':  np.mean(amp),
            'std': np.std(amp),
            'max': np.max(amp),
            'min': np. min(amp)
        })
    
    # 计算相对特征（比值）
    for k in range(cyc_period - 1):
        # 幅度比值
        mean_ratio = amp_stats[k+1]['mean'] / (amp_stats[k]['mean'] + 1e-10)
        std_ratio = amp_stats[k+1]['std'] / (amp_stats[k]['std'] + 1e-10)
        
        features.append(float(mean_ratio))
        features.append(float(std_ratio))
    
    # 归一化幅度分布特征
    all_amp = np. abs(sig_c)
    mean_amp = np. mean(all_amp) + 1e-10
    
    for k in range(cyc_period):
        index = slice(k * cyc_len, (k + 1) * cyc_len)
        seg_amp = np.abs(sig_c[index])
        
        # 归一化后的高阶统计量
        norm_amp = seg_amp / mean_amp
        if np.std(norm_amp) > 1e-10:
            skewness = float(np.mean((norm_amp - np.mean(norm_amp)) ** 3) / (np.std(norm_amp) ** 3))
            kurtosis = float(np. mean((norm_amp - np.mean(norm_amp)) ** 4) / (np.std(norm_amp) ** 4) - 3)
        else:
            skewness, kurtosis = 0.0, 0.0
        
        features.append(skewness)
        features.append(kurtosis)
    
    return features


def extract_higher_order_cumulants(sig:  np.ndarray) -> List[float]: 
    """
    高阶累积量特征 - 对高斯噪声不敏感
    提取C20, C21, C40, C41, C42等累积量
    """
    features = []
    sig = sig.flatten()
    
    # 归一化
    sig_norm = sig / (np.sqrt(np.mean(np. abs(sig)**2)) + 1e-10)
    
    # 二阶累积量
    C20 = np.mean(sig_norm ** 2)
    C21 = np.mean(np.abs(sig_norm) ** 2)
    
    # 四阶累积量
    C40 = np.mean(sig_norm ** 4) - 3 * C20 ** 2
    C41 = np. mean((sig_norm ** 3) * np.conj(sig_norm)) - 3 * C20 * C21
    C42 = np. mean(np.abs(sig_norm) ** 4) - np.abs(C20) ** 2 - 2 * C21 ** 2
    
    # 六阶累积量（可选，增加特征维度）
    C60 = np. mean(sig_norm ** 6) - 15 * np.mean(sig_norm ** 4) * C20 + 30 * C20 ** 3
    C63 = np.mean(np.abs(sig_norm) ** 6) - 9 * C42 * C21 - 6 * C21 ** 3
    
    features.extend([
        float(np.abs(C20)), float(np.angle(C20)),
        float(np.real(C21)),
        float(np.abs(C40)), float(np.angle(C40)),
        float(np.abs(C41)), float(np.angle(C41)),
        float(np.abs(C42)),
        float(np.abs(C60)), float(np.abs(C63))
    ])
    
    return features


def extract_iq_imbalance_features(sig: np. ndarray, cyc_period: int, cyc_len:  int) -> List[float]:
    """
    I/Q不平衡特征 - 设备硬件固有特性
    包括幅度不平衡和相位不平衡
    """
    features = []
    sig = sig.flatten()
    
    I = np.real(sig)
    Q = np.imag(sig)
    
    # 全局I/Q不平衡
    # 幅度不平衡
    I_power = np.mean(I ** 2)
    Q_power = np.mean(Q ** 2)
    amp_imbalance = (I_power - Q_power) / (I_power + Q_power + 1e-10)
    
    # 相位不平衡（I/Q正交性偏离）
    phase_imbalance = 2 * np.mean(I * Q) / (I_power + Q_power + 1e-10)
    
    features. append(float(amp_imbalance))
    features.append(float(phase_imbalance))
    
    # 每段的I/Q不平衡变化
    for k in range(cyc_period):
        index = slice(k * cyc_len, (k + 1) * cyc_len)
        seg = sig[index]
        
        I_seg = np. real(seg)
        Q_seg = np.imag(seg)
        
        I_p = np.mean(I_seg ** 2)
        Q_p = np.mean(Q_seg ** 2)
        
        seg_amp_imb = (I_p - Q_p) / (I_p + Q_p + 1e-10)
        seg_phase_imb = 2 * np.mean(I_seg * Q_seg) / (I_p + Q_p + 1e-10)
        
        features.append(float(seg_amp_imb))
        features.append(float(seg_phase_imb))
    
    return features


def extract_nonlinearity_features(sig: np.ndarray, cyc_period: int, cyc_len: int) -> List[float]:
    """
    非线性特征 - 功率放大器非线性特性
    """
    features = []
    sig = sig.flatten()
    
    amp = np.abs(sig)
    phase = np.angle(sig)
    
    # AM-AM非线性（幅度-幅度）
    if len(amp) > 1:
        # 使用多项式拟合检测非线性
        amp_normalized = amp / (np.max(amp) + 1e-10)
        
        # 计算AM-AM失真
        amp_diff = np.diff(amp_normalized)
        am_am_var = float(np.var(amp_diff))
        
        # 三阶非线性系数估计
        if np.std(amp_normalized) > 1e-10:
            third_order = float(np.mean(amp_normalized ** 3) / (np.mean(amp_normalized) ** 3 + 1e-10))
        else:
            third_order = 0.0
        
        features.append(am_am_var)
        features.append(third_order)
    else:
        features. extend([0.0, 0.0])
    
    # AM-PM非线性（幅度-相位）
    # 相位变化与幅度的相关性
    if len(amp) > 1:
        amp_diff = np.diff(amp)
        phase_unwrap = np.unwrap(phase)
        phase_diff = np. diff(phase_unwrap)
        
        # 幅度变化与相位变化的相关系数
        if np.std(amp_diff) > 1e-10 and np.std(phase_diff) > 1e-10:
            am_pm_corr = float(np.corrcoef(amp_diff, phase_diff)[0, 1])
            if np.isnan(am_pm_corr):
                am_pm_corr = 0.0
        else:
            am_pm_corr = 0.0
        
        features.append(am_pm_corr)
    else: 
        features.append(0.0)
    
    # 每段的非线性特征
    for k in range(min(cyc_period, 5)):  # 限制维度
        index = slice(k * cyc_len, (k + 1) * cyc_len)
        seg = sig[index]
        seg_amp = np. abs(seg)
        
        if len(seg_amp) > 1:
            # EVM-like特征
            seg_amp_norm = seg_amp / (np.mean(seg_amp) + 1e-10)
            evm = float(np.std(seg_amp_norm))
            features.append(evm)
        else:
            features.append(0.0)
    
    return features


# ============== 2. 特征对齐与域适应 ==============

class DomainAdapter:
    """
    域适应器 - 用于对齐源域和目标域的特征分布
    """
    
    def __init__(self, method: str = 'coral'):
        """
        Args:
            method:  域适应方法 ('coral', 'mmd', 'subspace')
        """
        self.method = method
        self.source_mean = None
        self.source_std = None
        self.source_cov = None
        self.transform_matrix = None
        self.pca = None
        self. scaler = StandardScaler()
    
    def fit(self, source_features: np.ndarray, source_labels: Optional[np.ndarray] = None):
        """
        在源域数据上拟合
        """
        # 标准化
        self.scaler.fit(source_features)
        source_scaled = self.scaler.transform(source_features)
        
        self.source_mean = np.mean(source_scaled, axis=0)
        self.source_std = np.std(source_scaled, axis=0) + 1e-10
        
        if self.method == 'coral':
            # CORAL:  计算源域协方差
            self.source_cov = np.cov(source_scaled. T) + np.eye(source_scaled.shape[1]) * 1e-6
        
        elif self.method == 'subspace':
            # 子空间对齐
            self.pca = PCA(n_components=min(50, source_scaled.shape[1]))
            self.pca.fit(source_scaled)
    
    def transform(self, target_features: np. ndarray) -> np.ndarray:
        """
        对目标域数据进行域适应变换
        """
        target_scaled = self. scaler.transform(target_features)
        
        if self.method == 'coral':
            return self._coral_transform(target_scaled)
        elif self.method == 'mmd':
            return self._mmd_transform(target_scaled)
        elif self.method == 'subspace':
            return self._subspace_transform(target_scaled)
        else:
            return target_scaled
    
    def _coral_transform(self, target_features: np.ndarray) -> np.ndarray:
        """
        CORAL (CORrelation ALignment) 变换
        对齐目标域和源域的二阶统计量
        """
        # 计算目标域协方差
        target_cov = np.cov(target_features.T) + np.eye(target_features.shape[1]) * 1e-6
        
        # 计算变换矩阵
        # target_cov^(-1/2) * source_cov^(1/2)
        try:
            # 特征分解
            U_t, S_t, _ = np.linalg.svd(target_cov)
            U_s, S_s, _ = np.linalg.svd(self.source_cov)
            
            # 计算平方根和逆平方根
            S_t_inv_sqrt = np.diag(1.0 / np.sqrt(S_t + 1e-10))
            S_s_sqrt = np.diag(np.sqrt(S_s + 1e-10))
            
            # 白化目标域
            target_whitened = target_features @ U_t @ S_t_inv_sqrt @ U_t.T
            
            # 着色为源域分布
            target_coral = target_whitened @ U_s @ S_s_sqrt @ U_s.T
            
            return target_coral
        except: 
            # 如果变换失败，返回标准化的特征
            return target_features
    
    def _mmd_transform(self, target_features: np.ndarray) -> np.ndarray:
        """
        简化的MMD对齐 - 均值和方差对齐
        """
        target_mean = np. mean(target_features, axis=0)
        target_std = np.std(target_features, axis=0) + 1e-10
        
        # 对齐均值和方差
        aligned = (target_features - target_mean) / target_std * self.source_std + self.source_mean
        
        return aligned
    
    def _subspace_transform(self, target_features: np. ndarray) -> np.ndarray:
        """
        子空间对齐
        """
        if self.pca is not None:
            return self.pca. transform(target_features)
        return target_features


# ============== 3. 数据增强 ==============

class RFDataAugmentor:
    """
    射频信号数据增强器 - 用于增加训练数据多样性
    """
    
    def __init__(self, noise_std: float = 0.01, phase_shift_max: float = 0.1,
                 freq_offset_max: float = 0.01, amp_scale_range:  Tuple[float, float] = (0.9, 1.1)):
        self.noise_std = noise_std
        self.phase_shift_max = phase_shift_max
        self.freq_offset_max = freq_offset_max
        self.amp_scale_range = amp_scale_range
    
    def augment(self, sig:  np.ndarray, num_augments: int = 5) -> List[np.ndarray]:
        """
        对单个信号进行多次增强
        """
        augmented = [sig. copy()]
        
        for _ in range(num_augments):
            aug_sig = sig.copy()
            
            # 随机选择增强类型
            aug_type = np.random.choice(['noise', 'phase', 'freq', 'amp', 'combined'])
            
            if aug_type == 'noise' or aug_type == 'combined': 
                aug_sig = self._add_noise(aug_sig)
            
            if aug_type == 'phase' or aug_type == 'combined': 
                aug_sig = self._add_phase_shift(aug_sig)
            
            if aug_type == 'freq' or aug_type == 'combined': 
                aug_sig = self._add_freq_offset(aug_sig)
            
            if aug_type == 'amp' or aug_type == 'combined': 
                aug_sig = self._scale_amplitude(aug_sig)
            
            augmented.append(aug_sig)
        
        return augmented
    
    def _add_noise(self, sig: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise_power = np.mean(np.abs(sig) ** 2) * self.noise_std ** 2
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*sig.shape) + 1j * np.random. randn(*sig. shape))
        return sig + noise
    
    def _add_phase_shift(self, sig: np.ndarray) -> np.ndarray:
        """添加随机相位偏移"""
        phase_shift = np.random.uniform(-self.phase_shift_max, self. phase_shift_max) * np.pi
        return sig * np.exp(1j * phase_shift)
    
    def _add_freq_offset(self, sig: np.ndarray) -> np.ndarray:
        """添加频率偏移"""
        n = len(sig. flatten())
        freq_offset = np.random.uniform(-self. freq_offset_max, self.freq_offset_max)
        t = np.arange(n)
        phase_ramp = np.exp(1j * 2 * np. pi * freq_offset * t)
        return sig. flatten() * phase_ramp
    
    def _scale_amplitude(self, sig: np.ndarray) -> np.ndarray:
        """随机缩放幅度"""
        scale = np.random.uniform(*self.amp_scale_range)
        return sig * scale


# ============== 4. 集成分类器 ==============

def build_robust_classifiers():
    """
    构建鲁棒性更强的分类器集合
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
    from sklearn. neural_network import MLPClassifier
    
    try:
        from lightgbm import LGBMClassifier
        has_lgbm = True
    except ImportError:
        has_lgbm = False
    
    classifiers = {
        # 使用更大的k值，提高泛化能力
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
        
        # SVM使用RBF核，添加正则化
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True),
        
        # 随机森林 - 减少过拟合
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', n_jobs=-1
        ),
        
        # 添加MLP神经网络
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', alpha=0.01, max_iter=500,
            early_stopping=True, validation_fraction=0.1
        ),
    }
    
    if has_lgbm:
        classifiers['LightGBM'] = LGBMClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
            verbose=-1
        )
    
    return classifiers


def create_ensemble_classifier():
    """
    创建集成分类器
    """
    from sklearn.ensemble import VotingClassifier
    
    base_classifiers = build_robust_classifiers()
    
    # 软投票集成
    ensemble = VotingClassifier(
        estimators=[(name, clf) for name, clf in base_classifiers.items() if hasattr(clf, 'predict_proba')],
        voting='soft'
    )
    
    return ensemble