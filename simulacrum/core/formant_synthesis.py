"""
共振峰声学合成器 — 共振峰→波形

将 vocalization.py 中 FormantSynthesizer 输出的共振峰频率
(F1, F2, F3) + F0(基频) + voicing 转换为可听的时域波形。

声源-滤波器模型 (Fant 1960):
  声源: 周期脉冲串(浊音) / 白噪声(清音)
  滤波器: 级联二阶共振器 (每个共振峰一个)
  辐射: +6dB/倍频程高通 (简化为简单预加重)

生物学对应:
  - 声带振动 → 周期脉冲源 (F0控制)
  - 咽腔+口腔+鼻腔共振 → 共振峰滤波
  - 唇辐射 → 高通预加重
  - 鼻音 → 额外鼻共振峰 + 反共振
"""

import numpy as np
import torch

# 默认参数
DEFAULT_SAMPLE_RATE = 22050   # Hz
DEFAULT_FRAME_MS = 10         # 每帧毫秒数
EPSILON = 1e-8


class FormantToWaveform:
    """
    共振峰到波形的声学合成器

    使用级联二阶 IIR 滤波器 (biquad) 实现共振峰滤波。
    声源使用 LF (Liljencrants-Fant) 模型的简化版本。
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        pre_emphasis: float = 0.97,
        n_formants: int = 3,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.samples_per_frame = int(sample_rate * frame_ms / 1000)
        self.pre_emphasis = pre_emphasis
        self.n_formants = n_formants

        # 共振峰带宽 (Hz) — 控制共振峰的"锐度"
        self.default_bandwidths = [80, 90, 120]  # F1窄, F2中, F3宽

    # ── 声源生成 ──

    def _glottal_pulse_train(
        self, f0: float, n_samples: int, sample_rate: int
    ) -> np.ndarray:
        """
        生成准周期脉冲串 (声源)

        简化 LF 模型: 模拟声带振动产生的气流脉冲。
        每个脉冲由上升沿和指数衰减下降沿组成。

        Args:
            f0: 基频 (Hz)
            n_samples: 输出采样数
            sample_rate: 采样率
        Returns:
            pulse: [n_samples] 浊音脉冲串
        """
        if f0 < 20:
            f0 = 20.0  # 最低可听基频

        pulse = np.zeros(n_samples, dtype=np.float64)
        period = int(sample_rate / f0)
        if period < 1:
            period = 1

        # LF 模型简化脉冲
        # 开相 (开相占周期的 40%)
        open_ratio = 0.40
        open_samples = int(period * open_ratio)
        close_samples = period - open_samples

        for i in range(0, n_samples, period):
            # 上升沿 (声带打开)
            for j in range(min(open_samples, n_samples - i)):
                t = j / max(open_samples, 1)
                # 升余弦上升
                pulse[i + j] = 0.5 * (1.0 - np.cos(np.pi * t))

            # 下降沿 (声带关闭，指数衰减)
            for j in range(open_samples, period):
                if i + j >= n_samples:
                    break
                t = (j - open_samples) / max(close_samples, 1)
                pulse[i + j] = np.exp(-3.0 * t) * 0.8

        return pulse

    def _noise_source(self, n_samples: int) -> np.ndarray:
        """生成白噪声声源 (清音)"""
        return np.random.randn(n_samples) * 0.3

    def _mixed_source(
        self, f0: float, voicing: float, n_samples: int
    ) -> np.ndarray:
        """
        混合声源: voicing 控制浊音/清音比例

        voicing=1.0 → 纯浊音 (脉冲串)
        voicing=0.0 → 纯清音 (噪声)
        0 < voicing < 1 → 混合 (如浊擦音 /h/, /v/)
        """
        pulse = self._glottal_pulse_train(f0, n_samples, self.sample_rate)
        noise = self._noise_source(n_samples)
        return voicing * pulse + (1.0 - voicing) * noise

    # ── 共振峰滤波器 ──

    def _biquad_coefficients(
        self, center_freq: float, bandwidth: float, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        计算二阶共振器 (peaking EQ) 的系数

        使用 Audio EQ Cookbook (Robert Bristow-Johnson) 公式。

        Args:
            center_freq: 共振峰中心频率 (Hz)
            bandwidth: 带宽 (Hz)
            sample_rate: 采样率

        Returns:
            b: [3] 前馈系数
            a: [3] 反馈系数
        """
        w0 = 2.0 * np.pi * center_freq / sample_rate
        alpha = np.sin(w0) * np.sinh(
            np.log(2.0) * bandwidth * w0 / (2.0 * np.sin(w0) + EPSILON)
        )

        # Peaking EQ (共振峰增益约 12dB)
        gain_db = 12.0
        A = 10 ** (gain_db / 40.0)

        cos_w0 = np.cos(w0)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        # 归一化
        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])

        return b, a

    def _apply_biquad(
        self, signal: np.ndarray, b: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """
        应用二阶 IIR 滤波器 (Direct Form I)

        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        """
        n = len(signal)
        output = np.zeros(n, dtype=np.float64)
        x1, x2 = 0.0, 0.0
        y1, y2 = 0.0, 0.0

        for i in range(n):
            x0 = signal[i]
            y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            output[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0

        return output

    # ── 辐射模型 ──

    def _radiation_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        简化辐射模型: +6dB/倍频程高通

        y[n] = x[n] - alpha * x[n-1]
        类似预加重，模拟唇部辐射的高频增强效应。
        """
        output = np.zeros_like(signal)
        output[0] = signal[0]
        for i in range(1, len(signal)):
            output[i] = signal[i] - self.pre_emphasis * signal[i - 1]
        return output

    # ── 主合成方法 ──

    def synthesize_frame(
        self,
        f0: float,
        formants_hz: np.ndarray,    # [n_formants] F1, F2, F3 (Hz)
        voicing: float = 1.0,
        intensity: float = 0.5,
        bandwidths: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        合成一帧音频

        Args:
            f0: 基频 (Hz)
            formants_hz: [n_formants] 共振峰频率
            voicing: 浊音度 [0,1]
            intensity: 响度 [0,1]
            bandwidths: 共振峰带宽 (Hz)，默认使用预设值

        Returns:
            audio: [samples_per_frame] 时域波形
        """
        n_samples = self.samples_per_frame

        # 1. 生成声源
        source = self._mixed_source(f0, voicing, n_samples)

        # 2. 级联共振峰滤波
        filtered = source.copy()
        for i in range(min(self.n_formants, len(formants_hz))):
            freq = float(formants_hz[i])
            bw = float(bandwidths[i]) if bandwidths is not None else self.default_bandwidths[i]

            # 频率有效性检查
            if freq < 50 or freq >= self.sample_rate / 2:
                continue

            b, a = self._biquad_coefficients(freq, bw, self.sample_rate)
            filtered = self._apply_biquad(filtered, b, a)

        # 3. 辐射滤波
        output = self._radiation_filter(filtered)

        # 4. 应用响度
        output *= intensity

        # 5. 归一化 (防止削波)
        peak = np.max(np.abs(output)) + EPSILON
        if peak > 1.0:
            output /= peak

        return output

    def synthesize(
        self,
        formants: np.ndarray,       # [T, n_formants] Hz
        f0: np.ndarray,              # [T, 1] 基频 (归一化 0~1)
        voicing: np.ndarray,         # [T, 1] 浊音概率
        intensity: float = 0.5,
        f0_range: tuple[float, float] = (80.0, 300.0),
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """
        合成完整音频

        Args:
            formants: [T, 3] 共振峰频率 (Hz)
            f0: [T, 1] 基频归一化值 [0,1]
            voicing: [T, 1] 浊音概率 [0,1]
            intensity: 整体响度 [0,1]
            f0_range: (min_f0, max_f0) Hz
            sample_rate: 采样率

        Returns:
            audio: [total_samples] float64 波形 [-1, 1]
        """
        sr = sample_rate or self.sample_rate

        # 确保 numpy
        if isinstance(formants, torch.Tensor):
            formants = formants.detach().cpu().numpy()
        if isinstance(f0, torch.Tensor):
            f0 = f0.detach().cpu().numpy()
        if isinstance(voicing, torch.Tensor):
            voicing = voicing.detach().cpu().numpy()

        formants = np.atleast_2d(formants)
        f0 = np.atleast_2d(f0)
        voicing = np.atleast_2d(voicing)

        # 确保 f0 和 voicing 有正确的列维度 (处理 1D 输入)
        if f0.ndim == 1 or f0.shape[0] == 1:
            f0 = f0.reshape(-1, 1)
        if voicing.ndim == 1 or voicing.shape[0] == 1:
            voicing = voicing.reshape(-1, 1)

        T = formants.shape[0]
        all_frames = []

        for t in range(T):
            # 反归一化 F0
            f0_val = float(f0[t, 0]) if f0.shape[1] > 0 else 0.5
            f0_hz = f0_range[0] + f0_val * (f0_range[1] - f0_range[0])

            # 共振峰
            frame_formants = formants[t]

            # 浊音度
            v = float(voicing[t, 0]) if voicing.shape[1] > 0 else 0.8

            # 合成一帧
            frame_audio = self.synthesize_frame(
                f0=f0_hz,
                formants_hz=frame_formants,
                voicing=v,
                intensity=intensity,
            )
            all_frames.append(frame_audio)

        # 拼接帧，添加帧间交叉淡入淡出以避免爆音
        audio = self._crossfade_frames(all_frames)

        return audio

    def _crossfade_frames(self, frames: list, fade_len: int = 32) -> np.ndarray:
        """
        帧间交叉淡入淡出拼接

        避免帧边界处的不连续导致的点击噪声。
        """
        if not frames:
            return np.array([], dtype=np.float64)

        if len(frames) == 1:
            return frames[0]

        result = frames[0].copy()
        fade_len = min(fade_len, self.samples_per_frame // 4)

        for i in range(1, len(frames)):
            current = frames[i]

            if fade_len > 0 and len(result) >= fade_len and len(current) >= fade_len:
                # 交叉淡入淡出
                fade_out = np.linspace(1.0, 0.0, fade_len)
                fade_in = np.linspace(0.0, 1.0, fade_len)

                result[-fade_len:] *= fade_out
                current[:fade_len] *= fade_in
                current[:fade_len] += result[-fade_len:]

                result = np.concatenate([result[:-fade_len], current])
            else:
                result = np.concatenate([result, current])

        return result

    # ── 文件输出 ──

    def save_wav(
        self,
        audio: np.ndarray,
        filepath: str,
        sample_rate: int | None = None,
    ) -> None:
        """
        保存为 WAV 文件

        使用 scipy.io.wavfile，如果不可用则用 wave 模块 (标准库)。
        """
        sr = sample_rate or self.sample_rate

        # 归一化到 int16 范围
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        try:
            from scipy.io import wavfile
            wavfile.write(filepath, sr, audio_int16)
        except ImportError:
            import wave
            with wave.open(filepath, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())

        print(f"[VOCAL] Saved audio: {filepath} ({len(audio)/sr:.2f}s, {sr}Hz)")


# ── 便捷工厂函数 ──

def create_formant_synthesizer(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
) -> FormantToWaveform:
    """创建共振峰波形合成器"""
    return FormantToWaveform(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
    )
