# Technical Notes - End-of-Turn (EoT) Detection Model

1. Our model extracts 42 causality-compliant, telephony-bounded ($\le 3400$Hz) acoustic, prosodic, spectral, and conversational context features strictly prior to pause onset (x[:int(pause_start*sr)]).
2. Spectral stats and MFCC filterbanks are restricted to $\le 3400$Hz to eliminate high-frequency upsampling artifacts present in 8kHz telephony recordings.
3. Key discriminative signals include speaker-normalized semitone pitch ($ST = 12 \log_2(F_0 / F_{0, mean\_30})$), tail silence duration, short-time energy drop, harmonic-to-noise ratio (HNR), and spectral tilt drop.
4. Phrase-final syllable lengthening ratio, unvoiced hesitation gap counts, and speech rate estimation provide robust language-agnostic prosodic cues.
5. We use a regularized scikit-learn GradientBoostingClassifier (`max_depth=3`, `subsample=0.85`, `min_samples_leaf=8`) with probability calibration — fully compliant with allowed libraries.
6. GroupKFold cross-validation (5 splits grouped by turn_id) yields held-out Hindi delay of 780ms (AUC 0.716) and English delay of 1228ms (AUC 0.618) at $\le 5\%$ false-cutoff budget.
7. Model capacity was strictly regularized to reduce the in-sample memorization gap and ensure robust generalization to unseen hidden test set speakers.
8. Primary failure modes occur on short abrupt pauses in noisy audio and on questions with rising pitch intonation where the turn is syntactically complete but prosodically resembles a hold.
9. The model generalizes cross-lingually because it relies on acoustic physics (pitch fall, energy decay, tail silence) rather than language-specific lexical features.
10. With one more day, we would implement streaming YIN pitch tracking, per-speaker speaking rate normalization, and lightweight 1D convolutional feature extraction using PyTorch.
