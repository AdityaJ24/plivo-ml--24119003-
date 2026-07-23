"""Advanced causality-compliant feature extraction for End-of-Turn (EoT) prediction.

CAUSALITY GUARANTEE:
For any pause starting at `pause_start`, ALL features are derived strictly from `x[0 : int(pause_start * sr)]`.
No future audio (samples after `pause_start`) or post-pause metadata (e.g. `pause_end`) is used.

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


def compute_pitch_autocorr(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.25):
    """Estimate fundamental frequency (F0 in Hz) using normalized autocorrelation."""
    frame = frame - np.mean(frame)
    max_val = np.max(np.abs(frame))
    if max_val < 1e-4:
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


def compute_mfcc_simple(fr, sr, n_mfcc=13, n_mels=26):
    """Compute simple MFCCs using numpy/scipy FFT for full dependency independence."""
    if len(fr) == 0:
        return np.zeros(n_mfcc, dtype=np.float32)

    fl = fr.shape[1]
    window = np.hamming(fl)
    fr_win = fr * window

    n_fft = 512
    spec = np.abs(scipy.fftpack.fft(fr_win, n=n_fft, axis=1)[:, :n_fft // 2 + 1])

    # Mel Filterbank
    low_freq = 0
    high_freq = sr / 2
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


def compute_spectral_stats(fr, sr):
    """Compute spectral centroid, rolloff, flux, and spectral tilt per frame."""
    if len(fr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    fl = fr.shape[1]
    window = np.hamming(fl)
    fr_win = fr * window
    n_fft = 512
    spec = np.abs(scipy.fftpack.fft(fr_win, n=n_fft, axis=1)[:, :n_fft // 2 + 1])
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

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

    # Spectral Tilt: ratio of low-freq energy (<1kHz) to high-freq energy (>1kHz)
    freq_1k_bin = int(1000 / (sr / 2) * (n_fft // 2))
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
    """Ratio of last voiced segment duration to average voiced segment duration.
    Final syllable lengthening is a strong cue for turn completion."""
    segments = find_voiced_segments(f0_contour_arr)
    if len(segments) < 2:
        return 1.0  # can't compare
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


# Total features: 62
N_FEATURES = 62


def extract_advanced_features(x, sr, pause_start, pause_index=0):
    """Extract a rich, causality-compliant feature vector strictly using x[0 : pause_start].

    Returns a feature vector of length N_FEATURES (62).
    """
    # Audio strictly before pause_start
    audio_full = x[:int(pause_start * sr)]
    if len(audio_full) < int(sr * 0.05):  # less than 50ms of audio
        return np.zeros(N_FEATURES, dtype=np.float32)

    # Extract trailing windows at multiple resolutions
    win_03 = get_speech_before(x, sr, pause_start, max_window_s=0.3)
    win_075 = get_speech_before(x, sr, pause_start, max_window_s=0.75)
    win_15 = get_speech_before(x, sr, pause_start, max_window_s=1.5)
    win_30 = get_speech_before(x, sr, pause_start, max_window_s=3.0)

    # Frames for windows
    fr_03 = compute_frames(win_03, sr)
    fr_075 = compute_frames(win_075, sr)
    fr_15 = compute_frames(win_15, sr)
    fr_30 = compute_frames(win_30, sr)

    # ======= 1. Energy Features (11) =======
    e_15 = compute_energy_db(fr_15)
    e_075 = compute_energy_db(fr_075)
    e_03 = compute_energy_db(fr_03)

    e_last = float(e_03[-1]) if len(e_03) > 0 else -100.0
    e_mean_03 = float(np.mean(e_03))
    e_mean_075 = float(np.mean(e_075))
    e_mean_15 = float(np.mean(e_15))
    e_std_075 = float(np.std(e_075)) if len(e_075) > 1 else 0.0
    e_max_15 = float(np.max(e_15))
    e_drop_ratio = e_last - e_mean_15  # energy drop in dB

    # Energy Slopes
    if len(e_03) >= 3:
        t_arr = np.arange(len(e_03))
        e_slope_03 = float(np.polyfit(t_arr, e_03, 1)[0])
    else:
        e_slope_03 = 0.0

    if len(e_075) >= 5:
        t_arr = np.arange(len(e_075))
        e_slope_075 = float(np.polyfit(t_arr, e_075, 1)[0])
    else:
        e_slope_075 = 0.0

    # Delta energy: frame-to-frame energy changes in last 300ms
    if len(e_03) > 1:
        delta_e = np.diff(e_03)
        e_delta_mean = float(np.mean(delta_e))
        e_delta_std = float(np.std(delta_e))
    else:
        e_delta_mean = 0.0
        e_delta_std = 0.0

    # ======= 2. Pitch / Prosody Features (14) =======
    f0_15 = compute_f0_contour(fr_15, sr)
    f0_075 = compute_f0_contour(fr_075, sr)
    f0_30 = compute_f0_contour(fr_30, sr)
    voiced_15 = f0_15[f0_15 > 0]
    voiced_075 = f0_075[f0_075 > 0]
    voiced_30 = f0_30[f0_30 > 0]

    voiced_ratio_03 = float(np.mean(f0_075[-5:] > 0)) if len(f0_075) >= 5 else 0.0
    voiced_ratio_075 = float(np.mean(f0_075 > 0)) if len(f0_075) > 0 else 0.0
    voiced_ratio_15 = float(np.mean(f0_15 > 0)) if len(f0_15) > 0 else 0.0

    if len(voiced_15) > 0:
        f0_mean_15 = float(np.mean(voiced_15))
        f0_std_15 = float(np.std(voiced_15)) if len(voiced_15) > 1 else 0.0
        f0_min_15 = float(np.min(voiced_15))
        f0_max_15 = float(np.max(voiced_15))
        f0_range_15 = f0_max_15 - f0_min_15
    else:
        f0_mean_15, f0_std_15, f0_min_15, f0_max_15, f0_range_15 = 0.0, 0.0, 0.0, 0.0, 0.0

    if len(voiced_075) >= 2:
        f0_last = float(voiced_075[-1])
        f0_drop_ratio = f0_last / (f0_mean_15 + 1e-6)
        t_voiced = np.arange(len(voiced_075))
        f0_slope_075 = float(np.polyfit(t_voiced, voiced_075, 1)[0])
    else:
        f0_last = 0.0
        f0_drop_ratio = 1.0
        f0_slope_075 = 0.0

    # Final syllable lengthening
    final_syl_ratio = compute_final_syllable_lengthening(f0_15)

    # Speech rate from 1.5s window
    speech_rate_15 = compute_speech_rate(f0_15)

    # ======= 3. Zero Crossing Rate (ZCR) Features (4) =======
    zcr_03 = compute_zcr(fr_03)
    zcr_075 = compute_zcr(fr_075)
    zcr_mean_03 = float(np.mean(zcr_03))
    zcr_mean_075 = float(np.mean(zcr_075))
    zcr_last = float(zcr_03[-1]) if len(zcr_03) > 0 else 0.0
    # ZCR slope
    if len(zcr_03) >= 3:
        zcr_slope = float(np.polyfit(np.arange(len(zcr_03)), zcr_03, 1)[0])
    else:
        zcr_slope = 0.0

    # ======= 4. Spectral Features (8) =======
    sc_mean_03, sroll_mean_03, sflux_03, stilt_03 = compute_spectral_stats(fr_03, sr)
    sc_mean_075, sroll_mean_075, sflux_075, stilt_075 = compute_spectral_stats(fr_075, sr)

    # ======= 5. Harmonic-to-Noise Ratio (1) =======
    hnr_075 = compute_hnr(fr_075, sr)

    # ======= 6. MFCC Summaries (20) =======
    mfcc_075 = compute_mfcc_simple(fr_075, sr, n_mfcc=13)
    if len(mfcc_075.shape) > 1 and mfcc_075.shape[0] > 0:
        mfcc_means = np.mean(mfcc_075, axis=0)
        mfcc_stds = np.std(mfcc_075, axis=0) if mfcc_075.shape[0] > 1 else np.zeros(13, dtype=np.float32)
    else:
        mfcc_means = np.zeros(13, dtype=np.float32)
        mfcc_stds = np.zeros(13, dtype=np.float32)

    # ======= 7. Temporal & Turn Context (4) =======
    turn_elapsed = float(pause_start)
    pause_idx = float(pause_index)
    # Pause position ratio: normalized pause index
    pause_pos_ratio = pause_idx / (pause_idx + 1.0)
    # Log of turn elapsed (compresses long turns)
    log_turn_elapsed = float(np.log1p(turn_elapsed))

    # ======= Assemble feature vector (62 features) =======
    features = np.hstack([
        # Energy (11)
        e_last, e_mean_03, e_mean_075, e_mean_15, e_std_075, e_max_15,
        e_drop_ratio, e_slope_03, e_slope_075, e_delta_mean, e_delta_std,
        # Pitch / Prosody (14)
        voiced_ratio_03, voiced_ratio_075, voiced_ratio_15,
        f0_mean_15, f0_std_15, f0_range_15,
        f0_last, f0_drop_ratio, f0_slope_075,
        f0_min_15, f0_max_15,
        final_syl_ratio, speech_rate_15,
        hnr_075,
        # ZCR (4)
        zcr_mean_03, zcr_mean_075, zcr_last, zcr_slope,
        # Spectral (8)
        sc_mean_03, sroll_mean_03, sflux_03, stilt_03,
        sc_mean_075, sroll_mean_075, sflux_075, stilt_075,
        # Context (4)
        turn_elapsed, pause_idx, pause_pos_ratio, log_turn_elapsed,
        # MFCC Means (10)
        mfcc_means[:10],
        # MFCC Stds (11)
        mfcc_stds[:11],
    ]).astype(np.float32)

    return features
