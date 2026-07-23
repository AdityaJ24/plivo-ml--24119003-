# Technical Notes - End-of-Turn (EoT) Detection Model

1. Our model extracts 78 multi-resolution acoustic, prosodic, spectral, and conversational context features strictly prior to pause onset (x[:int(pause_start*sr)]), with zero access to future samples or pause_end metadata.
2. Key discriminative signals include speaker-normalized semitone pitch (F0) trajectory, tail silence duration, short-time energy decay dynamics, harmonic-to-noise ratio (HNR), and spectral tilt ratio.
3. Semitone pitch rise indicators (for question intonation), preceding unvoiced hesitation gap counts, final syllable lengthening ratio, and speech rate estimation provide language-agnostic prosodic cues that improved Hindi performance to 770ms OOF delay and English performance to 1070ms OOF delay.
4. We use scikit-learn classifiers (ExtraTrees, GradientBoosting, HistGradientBoosting, MLPClassifier, RandomForest, LogisticRegression) in a soft VotingClassifier ensemble — fully compliant with the allowed library list.
5. GroupKFold cross-validation (5 splits grouped by turn_id) yields held-out AUC of 0.693 (English) and 0.774 (Hindi), with response delays of 1070ms and 770ms respectively at ≤5% false cutoff rate.
6. Top features by ExtraTrees importance are pause position ratio, tail silence duration, HNR, semitone pitch rise, spectral flux, final F0 value, and MFCC timbral coefficients — confirming that prosodic and spectral cues drive turn boundary detection.
7. Primary failure modes occur on short abrupt pauses in noisy audio and on questions with rising pitch intonation where the turn is syntactically complete but prosodically resembles a hold.
8. The model generalizes cross-lingually because it relies on acoustic physics (pitch fall, energy decay, tail silence) rather than language-specific lexical features.
9. With one more day, we would implement streaming YIN pitch tracking for more robust F0 estimation, add speaking rate normalization per speaker, and explore lightweight 1D convolutional feature extraction using PyTorch.
10. An additional improvement would be training separate per-language models with language-specific pitch range normalization and syllable timing features.
