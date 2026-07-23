"""Regularized, causality-compliant feature extraction for End-of-Turn (EoT) prediction.

CAUSALITY GUARANTEE:
For any pause starting at `pause_start`, ALL features are derived strictly from `x[0 : int(pause_start * sr)]`.
No future audio (samples after `pause_start`) or post-pause metadata (e.g. `pause_end`) is used.

BANDWIDTH BOUNDED:
FFT spectral stats and MFCC filterbanks are bounded to <= 3400 Hz to prevent overfitting to
telephony upsampling artifacts (4-8kHz quantization noise).

REDUNDANCY PRUNED:
Streamlined to 42 clean, non-redundant, language-agnostic prosodic and acoustic features.

LIBRARY COMPLIANCE:
Only uses numpy, scipy (allowed). No external pretrained models.
"""
import numpy as np
import scipy.signal
import scipy.fftpack


def load_wav(path):
    """Load 16kHz mono WAV file using scipy.io.wavfile (no soundfile dependency)."""
    import scipy.io.wavfile as wav
    sr, data = wav.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        x = data.astype(np.float32)
    return x, sr


def get_speech_before(x, sr, pause_start, max_window_s=3.0):
    """Extract audio strictly before pause_start up to max_window_s."""
    end = int(pause_start * sr)
    start = max(0, end - int(max_window_s * sr))
    return x[start:end]


def compute_frames(x, sr, frame_ms=25, hop_ms=10):
    """Slice 1D audio array into overlapping 2D frames."""
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def compute_energy_db(fr):
    """Compute Short-Time RMS Energy in dB for each frame."""
    if len(fr) == 0:
        return np.array([-100.0], dtype=np.float32)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def compute_zcr(fr):
    """Compute Zero Crossing Rate per frame."""
    if len(fr) == 0:
        return np.array([0.0], dtype=np.float32)
    signs = np.sign(fr)
    signs[signs == 0] = 1
    crossings = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    return crossings.astype(np.float32)


def compute_pitch_autocorr(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.20):
    """Estimate fundamental frequency (F0 in Hz) using normalized autocorrelation + energy check."""
    frame = frame - np.mean(frame)
    rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
    if rms < 0.005:  # quiet frame check
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def compute_f0_contour(fr, sr):
    """Compute F0 contour across frames."""
    if len(fr) == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array([compute_pitch_autocorr(f, sr) for f in fr], dtype=np.float32)


def compute_mfcc_telephony(fr, sr, n_mfcc=8, n_mels=20, max_freq=3400.0):
    """Compute MFCCs restricted to telephony bandwidth <= 3400 Hz."""
    if len(fr) == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)

    fl = fr.shape[1]
    window = np.hamming(fl)
    fr_win = fr * window

    n_fft = 512
    spec = np.abs(scipy.fftpack.fft(fr_win, n=n_fft, axis=1)[:, :n_fft // 2 + 1])

    # Mel Filterbank bounded to max_freq (3400 Hz)
    low_freq = 80.0
    high_freq = min(max_freq, sr / 2)
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_pts[m - 1]) / max(1, (bin_pts[m] - bin_pts[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_pts[m + 1] - k) / max(1, (bin_pts[m + 1] - bin_pts[m]))

    filter_banks = np.dot(spec, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT type II
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc


def compute_spectral_stats_telephony(fr, sr, max_freq=3400.0):
    """Compute spectral centroid, rolloff, flux, and spectral tilt restricted to telephony bandwidth <= 3400 Hz."""
    if len(fr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    fl = fr.shape[1]
    window = np.hamming(fl)
    fr_win = fr * window
    n_fft = 512
    max_bin = int(max_freq / (sr / 2) * (n_fft // 2)) + 1

    spec = np.abs(scipy.fftpack.fft(fr_win, n=n_fft, axis=1)[:, :max_bin])
    freqs = np.linspace(0, max_freq, max_bin)

    # Centroid
    sum_spec = np.sum(spec, axis=1) + 1e-12
    centroids = np.sum(spec * freqs, axis=1) / sum_spec

    # Rolloff (85%)
    cum_spec = np.cumsum(spec, axis=1)
    threshold = 0.85 * cum_spec[:, -1]
    rolloff_idx = np.array([np.where(row >= th)[0][0] if np.any(row >= th) else 0
                            for row, th in zip(cum_spec, threshold)])
    rolloffs = freqs[rolloff_idx]

    # Flux
    if len(spec) > 1:
        flux = np.mean(np.sqrt(np.sum(np.diff(spec, axis=0)**2, axis=1)))
    else:
        flux = 0.0

    # Spectral Tilt: ratio of energy below 1kHz vs 1kHz to 3.4kHz
    freq_1k_bin = int(1000 / max_freq * max_bin)
    low_energy = np.mean(np.sum(spec[:, :freq_1k_bin + 1] ** 2, axis=1))
    high_energy = np.mean(np.sum(spec[:, freq_1k_bin + 1:] ** 2, axis=1))
    spectral_tilt = float(np.log10(low_energy / (high_energy + 1e-12) + 1e-12))

    return float(np.mean(centroids)), float(np.mean(rolloffs)), float(flux), spectral_tilt


def compute_hnr(fr, sr):
    """Estimate Harmonic-to-Noise Ratio using autocorrelation peak."""
    if len(fr) == 0:
        return 0.0
    hnrs = []
    for frame in fr:
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-4:
            hnrs.append(0.0)
            continue
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if ac[0] <= 0:
            hnrs.append(0.0)
            continue
        ac = ac / ac[0]
        lo = int(sr / 400)
        hi = min(int(sr / 60), len(ac) - 1)
        if hi <= lo:
            hnrs.append(0.0)
            continue
        peak = np.max(ac[lo:hi])
        if peak <= 0 or peak >= 1.0:
            hnrs.append(0.0)
        else:
            hnrs.append(float(10 * np.log10(peak / (1 - peak + 1e-12) + 1e-12)))
    return float(np.mean(hnrs))


def find_voiced_segments(f0_contour_arr):
    """Find contiguous voiced segments (runs of nonzero F0). Returns list of (start_idx, length)."""
    segments = []
    in_segment = False
    start = 0
    for i, val in enumerate(f0_contour_arr):
        if val > 0:
            if not in_segment:
                start = i
                in_segment = True
        else:
            if in_segment:
                segments.append((start, i - start))
                in_segment = False
    if in_segment:
        segments.append((start, len(f0_contour_arr) - start))
    return segments


def compute_final_syllable_lengthening(f0_contour_arr, hop_ms=10):
    """Ratio of last voiced segment duration to average voiced segment duration."""
    segments = find_voiced_segments(f0_contour_arr)
    if len(segments) < 2:
        return 1.0
    lengths = [s[1] for s in segments]
    last_len = lengths[-1]
    avg_len = np.mean(lengths[:-1])
    if avg_len < 1:
        return 1.0
    return float(last_len / avg_len)


def compute_speech_rate(f0_contour_arr, hop_ms=10):
    """Estimate speech rate as voiced-to-unvoiced transitions per second."""
    if len(f0_contour_arr) < 2:
        return 0.0
    voiced = (f0_contour_arr > 0).astype(np.int32)
    transitions = np.sum(np.abs(np.diff(voiced)))
    duration_s = len(f0_contour_arr) * hop_ms / 1000.0
    if duration_s < 0.01:
        return 0.0
    return float(transitions / duration_s)


def compute_unvoiced_gaps(f0_contour_arr, hop_ms=10, min_gap_ms=50):
    """Count unvoiced gaps longer than min_gap_ms inside voiced speech."""
    min_gap_frames = int(min_gap_ms / hop_ms)
    voiced = (f0_contour_arr > 0).astype(np.int32)
    gaps = 0
    curr = 0
    first_v = np.where(voiced == 1)[0]
    if len(first_v) < 2:
        return 0
    start_idx, end_idx = first_v[0], first_v[-1]
    for val in voiced[start_idx:end_idx + 1]:
        if val == 0:
            curr += 1
        else:
            if curr >= min_gap_frames:
                gaps += 1
            curr = 0
    return gaps


# Total features: 42
N_FEATURES = 42


def extract_advanced_features(x, sr, pause_start, pause_index=0):
    """Extract a 42-dimensional, causality-compliant, band-limited (<=3400Hz) feature vector.

    Returns a feature vector of length N_FEATURES (42).
    """
    audio_full = x[:int(pause_start * sr)]
    if len(audio_full) < int(sr * 0.05):
        return np.zeros(N_FEATURES, dtype=np.float32)

    # Trailing windows
    win_015 = get_speech_before(x, sr, pause_start, max_window_s=0.15)
    win_03 = get_speech_before(x, sr, pause_start, max_window_s=0.3)
    win_075 = get_speech_before(x, sr, pause_start, max_window_s=0.75)
    win_15 = get_speech_before(x, sr, pause_start, max_window_s=1.5)
    win_30 = get_speech_before(x, sr, pause_start, max_window_s=3.0)

    fr_015 = compute_frames(win_015, sr)
    fr_03 = compute_frames(win_03, sr)
    fr_075 = compute_frames(win_075, sr)
    fr_15 = compute_frames(win_15, sr)
    fr_30 = compute_frames(win_30, sr)

    # ======= 1. Energy Features (15) =======
    e_30 = compute_energy_db(fr_30)
    e_15 = compute_energy_db(fr_15)
    e_075 = compute_energy_db(fr_075)
    e_03 = compute_energy_db(fr_03)
    e_015 = compute_energy_db(fr_015)

    e_last = float(e_03[-1]) if len(e_03) > 0 else -100.0
    e_mean_015 = float(np.mean(e_015)) if len(e_015) > 0 else -100.0
    e_mean_03 = float(np.mean(e_03))
    e_mean_075 = float(np.mean(e_075))
    e_mean_15 = float(np.mean(e_15))
    e_std_075 = float(np.std(e_075)) if len(e_075) > 1 else 0.0
    e_max_15 = float(np.max(e_15))
    e_drop_ratio = e_last - e_mean_15
    e_drop_100_300 = e_mean_015 - e_mean_03

    tail_silence_frames = 0
    for val in reversed(e_03):
        if val < -45.0:
            tail_silence_frames += 1
        else:
            break
    e_tail_silence_ms = float(tail_silence_frames * 10.0)

    e_slope_03 = float(np.polyfit(np.arange(len(e_03)), e_03, 1)[0]) if len(e_03) >= 3 else 0.0
    e_slope_075 = float(np.polyfit(np.arange(len(e_075)), e_075, 1)[0]) if len(e_075) >= 5 else 0.0

    delta_e = np.diff(e_03) if len(e_03) > 1 else np.array([0.0])
    e_delta_mean = float(np.mean(delta_e))
    e_delta_std = float(np.std(delta_e))

    # ======= 2. Pitch / Prosody Features (13) =======
    f0_15 = compute_f0_contour(fr_15, sr)
    f0_075 = compute_f0_contour(fr_075, sr)
    f0_30 = compute_f0_contour(fr_30, sr)
    f0_03 = compute_f0_contour(fr_03, sr)
    voiced_15 = f0_15[f0_15 > 0]
    voiced_075 = f0_075[f0_075 > 0]
    voiced_30 = f0_30[f0_30 > 0]
    voiced_03 = f0_03[f0_03 > 0]

    voiced_ratio_03 = float(np.mean(f0_075[-5:] > 0)) if len(f0_075) >= 5 else 0.0
    voiced_ratio_075 = float(np.mean(f0_075 > 0)) if len(f0_075) > 0 else 0.0
    voiced_ratio_15 = float(np.mean(f0_15 > 0)) if len(f0_15) > 0 else 0.0
    f0_mean_15 = float(np.mean(voiced_15)) if len(voiced_15) > 0 else 0.0
    f0_std_15 = float(np.std(voiced_15)) if len(voiced_15) > 1 else 0.0
    f0_last = float(voiced_075[-1]) if len(voiced_075) > 0 else 0.0
    f0_drop_ratio = f0_last / (f0_mean_15 + 1e-6)

    f0_mean_30 = float(np.mean(voiced_30)) if len(voiced_30) > 0 else f0_mean_15
    st_last = float(12.0 * np.log2(f0_last / f0_mean_30)) if (f0_last > 0 and f0_mean_30 > 0) else 0.0

    if len(voiced_03) >= 2 and len(voiced_15) > 0:
        st_min_15 = float(np.min(voiced_15))
        st_last_03 = float(voiced_03[-1])
        st_rise_indicator = float(12.0 * np.log2(st_last_03 / (st_min_15 + 1e-6))) if st_last_03 > 0 else 0.0
    else:
        st_rise_indicator = 0.0

    f0_slope_075 = float(np.polyfit(np.arange(len(voiced_075)), voiced_075, 1)[0]) if len(voiced_075) >= 2 else 0.0
    final_syl_ratio = compute_final_syllable_lengthening(f0_15)
    speech_rate_15 = compute_speech_rate(f0_15)
    unvoiced_gaps_15 = float(compute_unvoiced_gaps(f0_15))
    hnr_075 = compute_hnr(fr_075, sr)

    # ======= 3. Zero Crossing Rate (ZCR) Features (2) =======
    zcr_03 = compute_zcr(fr_03)
    zcr_mean_03 = float(np.mean(zcr_03))
    zcr_slope = float(np.polyfit(np.arange(len(zcr_03)), zcr_03, 1)[0]) if len(zcr_03) >= 3 else 0.0

    # ======= 4. Spectral Features (9, <= 3400 Hz Bounded) =======
    sc_mean_03, sroll_mean_03, sflux_03, stilt_03 = compute_spectral_stats_telephony(fr_03, sr)
    sc_mean_075, sroll_mean_075, sflux_075, stilt_075 = compute_spectral_stats_telephony(fr_075, sr)
    stilt_drop = stilt_03 - stilt_075

    # ======= 5. Temporal & Conversational Context (3) =======
    turn_elapsed = float(pause_start)
    pause_idx = float(pause_index)
    voiced_duration_tot = float(len(voiced_30) * 0.01)
    speech_to_elapsed_ratio = voiced_duration_tot / (turn_elapsed + 1e-3)

    # ======= Assemble 42-dimensional feature vector =======
    features = np.hstack([
        # Energy (15)
        e_last, e_mean_015, e_mean_03, e_mean_075, e_mean_15, e_std_075, e_max_15,
        e_drop_ratio, e_drop_100_300, e_tail_silence_ms, e_slope_03, e_slope_075,
        e_delta_mean, e_delta_std,
        # Pitch / Prosody (13)
        voiced_ratio_03, voiced_ratio_075, voiced_ratio_15, f0_mean_15, f0_std_15,
        f0_last, f0_drop_ratio, st_last, st_rise_indicator, f0_slope_075,
        final_syl_ratio, speech_rate_15, unvoiced_gaps_15, hnr_075,
        # ZCR (2)
        zcr_mean_03, zcr_slope,
        # Spectral (9)
        sc_mean_03, sroll_mean_03, sflux_03, stilt_03,
        sc_mean_075, sroll_mean_075, sflux_075, stilt_075, stilt_drop,
        # Context (3)
        turn_elapsed, pause_idx, speech_to_elapsed_ratio,
    ]).astype(np.float32)

    return features
