# Technical Notes - End-of-Turn (EoT) Detection Model

1. Our End-of-Turn (EoT) model extracts multi-resolution acoustic, prosodic, spectral, and conversational context features strictly prior to pause onset ($t \le \text{pause\_start}$).
2. Line-by-line causality auditing confirms audio is sliced as `x[:int(pause_start*sr)]` with zero access to future samples, and post-pause `pause_end` duration is never referenced.
3. Key signals include fundamental pitch ($F_0$) trajectory, pitch slope over final voiced speech, energy decay rate, RMS energy dynamics, zero-crossing rate (ZCR), spectral centroid/rolloff/flux, and MFCCs.
4. Pitch drop and energy decay serve as primary acoustic markers of terminal turn boundaries, whereas flat or rising pitch contours indicate continuation during hold pauses.
5. Terminal unvoiced phonemes and fricatives at pause boundaries are effectively distinguished using Zero-Crossing Rate (ZCR) dynamics and high-frequency spectral ratios.
6. Held-out GroupKFold cross-validation (5 splits grouped strictly by `turn_id`) yields out-of-fold (OOF) AUC of **0.685** (English) and **0.724** (Hindi), dropping English response delay by 424 ms vs baseline.
7. Refitting the ensemble on full data achieves an in-sample AUC of **0.988 / 0.996**, reflecting tree ensemble capacity, while GroupKFold OOF represents true unseen turn generalization.
8. Primary failure modes occur on short, abrupt pauses in noisy audio and on questions with rising pitch intonation where turn completion is syntactically present but prosodically resembles a hold.
9. With one more day, we would implement lightweight streaming pitch tracking (YIN/PYIN) with adaptive noise filtering to improve pitch contour estimation in low-SNR channels.
10. Finally, incorporating continuous speech rate estimation and CPU-friendly 1D depthwise-separable convolutional feature extraction would further refine cross-lingual generalization.
