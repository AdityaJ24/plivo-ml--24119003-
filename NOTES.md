# Technical Notes - End-of-Turn (EoT) Detection Model

1. Our End-of-Turn (EoT) model extracts multi-resolution acoustic, prosodic, spectral, and conversational context features strictly prior to pause onset ($t \le \text{pause\_start}$).
2. Key signals utilized include fundamental pitch ($F_0$) trajectory, pitch slope over final voiced speech, energy decay rate, RMS energy dynamics, zero-crossing rate (ZCR), spectral centroid, spectral rolloff, spectral flux, and MFCC coefficients.
3. Pitch drop and energy decay into the pause serve as primary acoustic markers of terminal turn boundaries, whereas flat or rising pitch contours indicate continuation during hold pauses.
4. Terminal unvoiced phonemes and fricatives at pause boundaries are effectively distinguished using Zero-Crossing Rate (ZCR) dynamics and high-frequency spectral ratios.
5. Conversational temporal context, such as total turn duration elapsed and pause sequence index, provides structural priors for turn completion probability.
6. The primary failure mode occurs on short, abrupt pauses where the speaker's vocal pitch remains flat or cuts off suddenly due to background ambient noise or telephony bandpass compression.
7. Additional misclassifications occur on questions with rising pitch intonation where turn completion is syntactically present but prosodically mirrors a continuation hold.
8. With one more day, we would implement lightweight streaming pitch tracking with adaptive noise filtering to improve pitch contour estimation in noisy channels.
9. Furthermore, we would incorporate continuous speech rate estimation and energy acceleration deltas to capture subtle micro-prosodic phrasing dynamics before pause onset.
10. Finally, exploring CPU-friendly lightweight 1D depthwise-separable convolutional networks combined with multi-task cross-lingual domain adaptation would further refine generalization.
