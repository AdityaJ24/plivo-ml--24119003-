"""Advanced causality-compliant feature extraction for End-of-Turn (EoT) prediction.

CAUSALITY GUARANTEE:
For any pause starting at `pause_start`, ALL features are derived strictly from `x[0 : int(pause_start * sr)]`.
No future audio (samples after `pause_start`) or post-pause metadata (e.g. `pause_end`) is used.
"""
import numpy as np
import scipy.signal
import scipy.fftpack


def load_wav(path):
    """Load 16kHz mono WAV file using scipy.io.wavfile (robust against missing soundfile)."""
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
        return np.array([ -100.0 ], dtype=np.float32)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def compute_zcr(fr):
    """Compute Zero Crossing Rate per frame."""
    if len(fr) == 0:
        return np.array([0.0], dtype=np.float32)
    signs = np.sign(fr)
    # Replace zeros with 1 to avoid artificial crossings
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
    
    # Hamming window
    fl = fr.shape[1]
    window = np.hamming(fl)
    fr_win = fr * window
    
    # Magnitude Spectrum
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
    """Compute spectral centroid, rolloff, and flux per frame."""
    if len(fr) == 0:
        return 0.0, 0.0, 0.0
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
    rolloff_idx = np.array([np.where(row >= th)[0][0] if np.any(row >= th) else 0 for row, th in zip(cum_spec, threshold)])
    rolloffs = freqs[rolloff_idx]
    
    # Flux
    if len(spec) > 1:
        flux = np.mean(np.sqrt(np.sum(np.diff(spec, axis=0)**2, axis=1)))
    else:
        flux = 0.0
        
    return float(np.mean(centroids)), float(np.mean(rolloffs)), float(flux)


def extract_advanced_features(x, sr, pause_start, pause_index=0):
    """Extract a rich, causality-compliant feature vector strictly using x[0 : pause_start]."""
    # Audio strictly before pause_start
    audio_full = x[:int(pause_start * sr)]
    if len(audio_full) < int(sr * 0.05): # less than 50ms of audio
        return np.zeros(48, dtype=np.float32)
    
    # Extract trailing windows
    win_03 = get_speech_before(x, sr, pause_start, max_window_s=0.3)
    win_075 = get_speech_before(x, sr, pause_start, max_window_s=0.75)
    win_15 = get_speech_before(x, sr, pause_start, max_window_s=1.5)
    
    # Frames for windows
    fr_03 = compute_frames(win_03, sr)
    fr_075 = compute_frames(win_075, sr)
    fr_15 = compute_frames(win_15, sr)
    fr_full = compute_frames(audio_full, sr)
    
    # 1. Energy Features
    e_15 = compute_energy_db(fr_15)
    e_075 = compute_energy_db(fr_075)
    e_03 = compute_energy_db(fr_03)
    
    e_last = e_03[-1] if len(e_03) > 0 else -100.0
    e_mean_03 = float(np.mean(e_03))
    e_mean_075 = float(np.mean(e_075))
    e_mean_15 = float(np.mean(e_15))
    e_std_075 = float(np.std(e_075)) if len(e_075) > 1 else 0.0
    e_max_15 = float(np.max(e_15))
    e_drop_ratio = e_last - e_mean_15  # energy drop in dB
    
    # Energy Slope over last 300ms (linear regression slope)
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
        
    # 2. Pitch / Prosody Features
    f0_15 = compute_f0_contour(fr_15, sr)
    f0_075 = compute_f0_contour(fr_075, sr)
    voiced_15 = f0_15[f0_15 > 0]
    voiced_075 = f0_075[f0_075 > 0]
    
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
        # Pitch slope over last voiced segment
        t_voiced = np.arange(len(voiced_075))
        f0_slope_075 = float(np.polyfit(t_voiced, voiced_075, 1)[0])
    else:
        f0_last = 0.0
        f0_drop_ratio = 1.0
        f0_slope_075 = 0.0

    # 3. Zero Crossing Rate (ZCR) Features
    zcr_03 = compute_zcr(fr_03)
    zcr_075 = compute_zcr(fr_075)
    zcr_mean_03 = float(np.mean(zcr_03))
    zcr_mean_075 = float(np.mean(zcr_075))
    zcr_last = float(zcr_03[-1]) if len(zcr_03) > 0 else 0.0
    
    # 4. Spectral Features (Centroid, Rolloff, Flux)
    sc_mean_03, sroll_mean_03, sflux_03 = compute_spectral_stats(fr_03, sr)
    sc_mean_075, sroll_mean_075, sflux_075 = compute_spectral_stats(fr_075, sr)
    
    # 5. MFCC Summaries (Mean & Std over last 0.75s)
    mfcc_075 = compute_mfcc_simple(fr_075, sr, n_mfcc=13)
    if len(mfcc_075) > 0:
        mfcc_means = np.mean(mfcc_075, axis=0)
        mfcc_stds = np.std(mfcc_075, axis=0) if len(mfcc_075) > 1 else np.zeros(13, dtype=np.float32)
    else:
        mfcc_means = np.zeros(13, dtype=np.float32)
        mfcc_stds = np.zeros(13, dtype=np.float32)

    # 6. Temporal & Turn Context
    turn_elapsed = float(pause_start)
    pause_idx = float(pause_index)
    
    # Assemble feature vector (48 features)
    features = np.hstack([
        # Energy (9)
        e_last, e_mean_03, e_mean_075, e_mean_15, e_std_075, e_max_15, e_drop_ratio, e_slope_03, e_slope_075,
        # Pitch / Prosody (11)
        voiced_ratio_03, voiced_ratio_075, voiced_ratio_15, f0_mean_15, f0_std_15, f0_range_15, f0_last, f0_drop_ratio, f0_slope_075, f0_min_15, f0_max_15,
        # ZCR & Spectral (6)
        zcr_mean_03, zcr_mean_075, zcr_last, sc_mean_03, sroll_mean_03, sflux_075,
        # Context (2)
        turn_elapsed, pause_idx,
        # MFCC Means (10) - first 10 coefs
        mfcc_means[:10],
        # MFCC Stds (10) - first 10 coefs
        mfcc_stds[:10],
    ]).astype(np.float32)
    
    return features
