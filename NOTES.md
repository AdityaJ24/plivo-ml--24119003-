# Technical Notes - End-of-Turn (EoT) Detection Model

1. Our model extracts 62 multi-resolution acoustic, prosodic, spectral, and conversational context features strictly prior to pause onset (x[:int(pause_start*sr)]), with zero access to future samples or pause_end metadata.
2. Key discriminative signals include fundamental pitch (F0) trajectory and terminal slope, short-time energy decay dynamics (including frame-to-frame delta energy), harmonic-to-noise ratio (HNR), and spectral tilt ratio.
3. Final syllable lengthening ratio and speech rate estimation (voiced-to-unvoiced transitions per second) provide language-agnostic prosodic cues that improved Hindi performance from baseline 850ms to 786ms OOF delay.
4. We use only scikit-learn classifiers (ExtraTrees, GradientBoosting, HistGradientBoosting, RandomForest, LogisticRegression) in a soft VotingClassifier ensemble — fully compliant with the allowed library list.
5. GroupKFold cross-validation (5 splits grouped by turn_id) yields held-out AUC of 0.663 (English) and 0.764 (Hindi), with response delays of 1195ms and 786ms respectively at ≤5% false cutoff rate.
6. Top features by ExtraTrees importance are pause position ratio, HNR, spectral flux, final F0 value, and MFCC timbral coefficients — confirming that prosodic and spectral cues drive turn boundary detection.
7. Primary failure modes occur on short abrupt pauses in noisy audio and on questions with rising pitch intonation where the turn is syntactically complete but prosodically resembles a hold.
8. The model generalizes cross-lingually because it relies on acoustic physics (pitch fall, energy decay) rather than language-specific lexical features.
9. With one more day, we would implement streaming YIN pitch tracking for more robust F0 estimation, add speaking rate normalization per speaker, and explore lightweight 1D convolutional feature extraction using PyTorch.
10. An additional improvement would be training separate per-language models with language-specific pitch range normalization and syllable timing features.
