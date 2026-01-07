import numpy as np


def get_ltf_template_and_mask(bandwidth='20M'):
    """
    返回标准 LTF 频域模板（64 点）和有效子载波掩码
    bandwidth: '20M', '10M', or '5M'
    """
    # 全带宽 LTF 值（±1~±26）
    full_vals = np.array([
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
        1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1,  # +1 ~ +26
        1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1  # -26 ~ -1
    ]) * np.sqrt(13 / 6)

    X_ltf = np.zeros(64, dtype=complex)
    mask = np.zeros(64, dtype=bool)

    if bandwidth == '20M':
        # ±1~±26
        X_ltf[33:59] = full_vals[:26]
        X_ltf[6:32] = full_vals[26:]
        mask[6:32] = True
        mask[33:59] = True
    elif bandwidth == '10M':
        # ±1~±13
        X_ltf[33:46] = full_vals[:13]  # +1~+13
        X_ltf[19:32] = full_vals[39:52]  # -13~-1 (注意顺序)
        mask[19:32] = True
        mask[33:46] = True
    elif bandwidth == '5M':
        # ±1~±6
        X_ltf[33:39] = full_vals[:6]  # +1~+6
        X_ltf[26:32] = full_vals[46:52]  # -6~-1
        mask[26:32] = True
        mask[33:39] = True
    else:
        raise ValueError("bandwidth must be '20M', '10M', or '5M'")

    return X_ltf, mask


def estimate_channel_from_ltf(rx_preamble, bandwidth='20M'):
    """从 320 点前导估计 64 点频域信道，带宽自适应"""
    ltf1 = rx_preamble[160:240]  # 80 pts
    ltf2 = rx_preamble[240:320]  # 80 pts
    ltf1_no_cp = ltf1[16:]  # 64
    ltf2_no_cp = ltf2[16:]  # 64

    Y1 = np.fft.fft(ltf1_no_cp, 64)
    Y2 = np.fft.fft(ltf2_no_cp, 64)

    X_ltf, valid_mask = get_ltf_template_and_mask(bandwidth)

    H1 = np.zeros(64, dtype=complex)
    H2 = np.zeros(64, dtype=complex)
    H1[valid_mask] = Y1[valid_mask] / X_ltf[valid_mask]
    H2[valid_mask] = Y2[valid_mask] / X_ltf[valid_mask]

    H_est = (H1 + H2) / 2.0

    # 可选：对无效子载波插值（这里简单用邻域平均，或保持为0）
    # 更高级可用线性/spline 插值，此处暂不处理
    return H_est


def extend_channel_to_N(H_64, N=320):
    """将 64 点频域信道扩展到 N 点（中心对齐）"""
    H_N = np.zeros(N, dtype=complex)
    start = (N - 64) // 2
    H_N[start:start + 64] = H_64
    return H_N


def equalize_full_preamble_adaptive(rx_preamble, bandwidth='20M'):
    """
    带宽自适应的全前导均衡（输入输出均为 320 点）

    Parameters:
        rx_preamble: (320,) complex
        bandwidth: '20M', '10M', or '5M'

    Returns:
        x_eq: (320,) complex — 均衡后的时域前导
    """
    if len(rx_preamble) != 320:
        raise ValueError("Input must be exactly 320 points")

    # Step 1: 带宽自适应信道估计
    H_64 = estimate_channel_from_ltf(rx_preamble, bandwidth=bandwidth)

    # Step 2: 扩展到 320 点频域
    H_320 = extend_channel_to_N(H_64, N=320)

    # Step 3: 对整个 320 点做 FFT
    Y_320 = np.fft.fft(rx_preamble, n=320)

    # Step 4: 频域 ZF 均衡
    eps = 1e-6
    X_eq_320 = Y_320 / (H_320 + eps)

    # Step 5: IFFT 回时域
    x_eq = np.fft.ifft(X_eq_320) * 320  # 补偿 NumPy IFFT 的 1/N 缩放

    return x_eq


# -----------------------------
# 示例：测试三种带宽
# -----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 模拟发送信号（以 20M 为例）
    X_ltf, _ = get_ltf_template_and_mask('20M')
    ltf_time = np.fft.ifft(X_ltf) * 64
    cp = ltf_time[-16:]
    ltf_with_cp = np.concatenate([cp, ltf_time])
    stf_part = np.random.randn(160) + 1j * np.random.randn(160)
    tx_preamble = np.concatenate([stf_part, ltf_with_cp, ltf_with_cp])

    # 模拟信道
    channel = np.array([1.0, 0.4 + 0.3j])
    rx_preamble = np.convolve(tx_preamble, channel)[:320]

    # 加噪声
    snr_db = 15
    noise_power = np.var(rx_preamble) / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(320) + 1j * np.random.randn(320))
    rx_preamble += noise

    # 测试不同带宽假设下的均衡
    for bw in ['20M', '10M', '5M']:
        x_eq = equalize_full_preamble_adaptive(rx_preamble, bandwidth=bw)
        print(f"Bandwidth={bw}, output shape={x_eq.shape}")

    # 可视化 20M 均衡效果
    x_eq_20M = equalize_full_preamble_adaptive(rx_preamble, bandwidth='20M')
    plt.figure(figsize=(12, 4))
    plt.plot(np.real(tx_preamble), label='Original TX', alpha=0.7)
    plt.plot(np.real(rx_preamble), label='Received', alpha=0.7)
    plt.plot(np.real(x_eq_20M), label='Equalized (20M)', linewidth=2)
    plt.legend()
    plt.title('Full Preamble Equalization (320 points)')
    plt.grid(True)
    plt.show()