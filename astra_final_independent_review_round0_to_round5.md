**Final Independent Review: Francisv2, Rounds 0–5**

Audit date: 1 November 2026. Workspace: `C:\Users\fpike\VoiceCreation`. Analysis runtime: `C:\GPT-SoVITS\runtime\python.exe`.

**Verdict**

Round 5 resolves the targeted Round 4 preservation failures on the tested material. It provides convincing controlled evidence of better consonant and pitch preservation while retaining most of the measured voiced de-fuzzing. My provisional engineering score is **7.75/10**, versus the pipeline's reproduced **7.19/10**.

I recommend Round 5 as the technical successor to Round 4. I do **not** certify it as >=8.5/10 studio quality. No auditory listening test was performed; the remaining uncertainty includes perceived naturalness, accent, identity, and the audibility of processing artifacts. A numerical engineering grade is not listening-derived MOS, and its two decimal places are display formatting rather than measurement precision.

**Scope, independence, and reproducibility**

I independently measured all six final renders, the saved Round 5 pre-master, and all four saved intermediate stages. Measurements use the original independent audit's acoustic routines, not `evaluate_quality.py`: Praat AC pitch/harmonicity, separate NumPy/SciPy spectra, Burg formants, oversampled peaks, and FFmpeg loudness. I added Praat cross-correlation as a second pitch detector, so preservation is not assessed solely with the same AC estimator now used inside the DSP. These are independently run diagnostics, not human ground truth.

All Round 0–4 WAV hashes still match the first audit. Every Round 5 saved stage regenerates exactly, sample for sample at PCM16 resolution, from the saved pre-master using the current DSP. The public `master_synthesized_voice` function and a mocked-HTTP `voice_agent.speak_realtime` harness also produce the exact saved final PCM16 waveform from that same input. The deployed `samples/francisv2_sample.wav` is byte-identical to Round 5.

The original field recording was decoded and spectrally inspected over 1698.88 seconds of complete analysis frames. References remain conditioned field recordings, not a clean studio target. The previous verified training manifest contains 75 clips totaling 372.77 seconds.

Production files, weights, and renders were not modified. New audit scripts, results, plots, a local client-harness WAV, and this report were added. The original review and original audit results were preserved.

**Independent score table**

The unchanged rubric equally weights speech-band artifact control, useful spectral balance, pitch/envelope integrity, and level/dynamics/pause handling. No reward is given for sample rate alone or added digital silence. Scores are evidence-informed engineering judgments; differences below about one point are not established perceptual differences.

| Round | Independent provisional score /10 | Pipeline score, reproduced | Assessment |
|---|---:|---:|---|
| 0 — Baseline | **5.00** | 4.99 | Weak presence and weak harmonic separation |
| 1 — Curated/conditioned references | **6.00** | 6.70 | Improved presence; limited independent HF de-fuzz evidence |
| 2 — Retrained v2ProPlus | **6.75** | 6.95 | Better mid-band structure; persistent source-bandwidth limitation |
| 3 — Low-noise prior + AP-BWE | **6.50** | 6.84 | Some harmonic improvement; discontinuous upper spectrum |
| 4 — Initial Studio Master | **7.00** | 8.60 | Strong de-fuzzing with preservation defects and metric inflation |
| 5 — Remediated master | **7.75** | 7.19 | Preservation fixes verified; studio-quality claim remains unvalidated |

Round 5's component grades are 8.0, 7.0, 8.5, and 7.5, respectively. Earlier independent grades are retained. All six published pipeline totals were rerun and matched exactly; agreement demonstrates reproducibility, not validity of the total as a quality scale.

**Independent acoustic comparison**

| Measurement | R0 | R1 | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|---:|
| Duration, seconds | 11.50 | 10.08 | 8.62 | 11.06 | 10.36 | 9.41 |
| Sample rate, kHz | 32 | 32 | 32 | 48 | 48 | 48 |
| Median Praat HNR, dB | 5.17 | 4.10 | 5.30 | 5.83 | 6.30 | 5.72 |
| Median confident F0, Hz | 119.7 | 144.7 | 137.2 | 161.7 | 150.8 | 142.5 |
| 0.5–4 kHz valley/harmonic density, dB | −6.96 | −9.25 | −11.14 | −12.70 | −16.99 | −16.98 |
| 4–8 kHz valley/harmonic density, dB | −0.37 | −0.48 | −0.58 | −0.69 | −4.45 | −3.98 |
| Integrated loudness, LUFS | −20.7 | −22.5 | −21.5 | −20.4 | −19.5 | −20.9 |
| Active-speech crest factor, dB | 16.88 | 18.08 | 20.33 | 19.26 | 18.69 | 19.90 |
| Estimated 4x true peak, dBTP | −3.34 | −4.47 | −1.38 | −1.11 | −1.08 | −1.05 |
| Leading/trailing exact zeros, seconds | 0 / .30 | 0 / .28 | 0 / .25 | 0 / 0 | .35 / .35 | 0 / 0 |

All six final renders have zero full-scale samples. The EQ intermediate is a separate exception described below.

HNR is the median of valid Praat frames, not a matched-vowel laboratory HNR. Valley/harmonic ratios use stationary, confidently voiced frames: more-negative values indicate deeper harmonic valleys, which can result from helpful noise suppression or artificially imposed periodicity. Absolute cross-round differences are confounded by different synthesis timing, pitch, references, and inference choices. The Round 5 stage comparisons provide the stronger causal evidence.

**Round-by-round findings**

**Round 0 — 5.00/10.** Weak upper-midrange presence and the shallowest measured harmonic valleys establish a reasonable baseline for improvement. Active 2–4 kHz energy is −17.94 dB relative to 80 Hz–16 kHz energy. Headroom is adequate, and confidently tracked pitch shows no adjacent steps exceeding three semitones. The principal limitations are spectral imbalance and relatively unstructured voiced HF energy; natural breathiness and vocoder fuzz cannot be fully separated without listening or a matched clean reference.

**Round 1 — 6.00/10.** Active 2–4 kHz energy improves to −13.67 dB, and mid-band valley separation improves by about 2.3 dB. The conditioning/reference changes improve spectral presentation. However, 4–8 kHz separation barely changes, and median Praat HNR falls rather than rises. This prevents a claim of uniform acoustic improvement. The Wiener processing affects dataset/reference preparation; the Round 1 runner saves the synthesized waveform directly. Retraining belongs to Round 2.

**Round 2 — 6.75/10.** Training records support the 75-clip v2ProPlus retraining. Mid-band separation improves further, and confidently tracked pitch remains stable. The shorter 8.62-second delivery and 20.33 dB active crest factor merit listening checks for pacing and transient balance. The upper spectrum remains sharply limited, and HF valley structure is still close to baseline. ASR confidence does not establish accurate pronunciation of the greeting, model name, or proper nouns.

**Round 3 — 6.50/10.** HNR and mid-band separation improve, but the spectrum has a strong 8–12 kHz deficit followed by renewed 12–16 kHz energy. That makes the BWE output weak evidence of natural bandwidth restoration. The nominal 48 kHz rate does not solve the inherited narrow source bandwidth. References, inference noise, sampling settings, and timing change together, so AP-BWE and the low-noise prior cannot be isolated from the archived render alone.

**Round 4 — 7.00/10.** Voiced inter-harmonic suppression improves materially, including in 4–8 kHz. But forced comb processing on unvoiced frames, harmonic peak sharpening, aggressive gating on some inputs, and zero-padding-driven score inflation limit the result. The prior same-input tests found unvoiced-to-voiced reclassification of 19.69% on Round 2 and 25.51% on the primary reference. The old reference-input gate test attenuated 13.27% of active unvoiced frames by more than 6 dB. Its final WAV is unclipped, but its original −16 LUFS claim was not met.

**Round 5 — 7.75/10.** It retains approximately Round 4-level harmonic valley separation while eliminating the measured large pitch and unvoiced-preservation failures on the test set. Its no-padding, gentler processing and reproducible stage exports are substantial engineering improvements. Remaining limitations are inherited bandwidth, unvalidated perceptual quality, modest redistribution of voiced/unvoiced balance, approximate loudness metering, and several packaging/reproducibility details. The legacy score falling from 8.60 to 7.19 does not negate the preservation evidence; the legacy score still rewards the earlier confounds.

**Round 5 stage-by-stage evidence**

| Stage | HNR median, dB | 0.5–4 kHz valley ratio, dB | 4–8 kHz valley ratio, dB | Loudness, LUFS |
|---|---:|---:|---:|---:|
| Saved trimmed 32 kHz pre-master | 5.55 | −12.17 | −0.58 | −21.1 |
| 48 kHz resampling + high-pass | 5.34 | −12.33 | −0.52 | −21.1 |
| One comb pass | 5.62 | −15.23 | −2.24 | −20.4 |
| Two comb passes | 5.80 | −16.97 | −3.85 | −19.9 |
| Comb + EQ | 5.71 | −16.98 | −3.94 | −19.8 |
| Final gate/level/peak stage | 5.72 | −16.98 | −3.98 | −20.9 |

The second pass adds roughly 1.7 dB of mid-band and 1.6 dB of HF valley separation. It also increases harmonic emphasis and consumes peak headroom. One versus two passes should still be compared by loudness-matched listening; a lower valley ratio is not automatically preferable.

**Were the specific Round 4 defects resolved?**

| Check | Follow-up result | Verdict |
|---|---|---|
| `original_unvoiced_now_confidently_voiced_pct` | **0/360 = 0%** for Round 5, including both comb stages and final output | Resolved on tested utterance |
| `matched_pitch_shift_over_3_semitones_pct` | **0%**, with 429 matched confident frames at final output | Resolved on tested utterance |
| Gate attenuation over 6 dB in active unvoiced frames | **0/352 = 0%** in the saved EQ-to-final comparison | Resolved on tested utterance |
| Outer zero padding | **0 seconds** at both ends; no added padding in new DSP | Removed |
| Hidden offline `noise_scale=0.45` | Removed; inspected shared decoder default is **0.5** | Specific noise-scale mismatch resolved |
| Exact offline/client equivalence | Same-input mocked client output is exact; runner alone trims ends | Substantially improved, not complete live parity proof |

The second detector agrees: cross-correlation finds zero newly voiced frames among 357 originally unvoiced frames and zero >3-semitone shifts in the saved final comparison. This supports robustness beyond the DSP's own AC tracker.

I also applied the old committed Round 4 comb and the new comb to the exact same Round 5 high-passed input. The old comb produces **85/360 = 23.61%** new confident voicing and **3.94%** >3-semitone matched-pitch shifts; the new comb produces **0%** for both. Cross-correlation reports 27.17% and 4.78% for the old comb and zero for the new one. This is stronger evidence than comparing unrelated renders.

Cross-input checks using Round 2 and the primary reference likewise produce zero new confident voicing, zero large matched-pitch shifts, and zero >6 dB gate attenuation under both detectors with the new chain.

**Preservation is improved, not mathematically exact**

After two comb passes on Round 5, median broad-envelope shape error is **0.302 dB**, versus **0.552 dB** for the old comb on the same input. Median absolute Burg F1/F2/F3 shifts are **1.98 / 5.07 / 6.67 Hz**; the F3 90th percentile is **38.36 Hz**, versus **87.56 Hz** with the old comb. After EQ and finalization, median shifts are approximately **2.42 / 5.68 / 8.18 Hz**. These are small, nonzero tracker changes; they are not proof of changed physical formants or audible identity loss.

Median final matched-pitch change is **0.0096 semitones**, and two of 431 initially confident AC frames no longer meet the matched-confidence criterion. Thus “0% above three semitones” must not be rewritten as “F0 invariant.”

The absolute median gain of originally unvoiced frames through the two comb passes is effectively 0 dB, but the largest measured frame change is **0.86 dB**. Voiced overlap and STFT reconstruction affect boundary windows. The median voiced-frame gain is about **+1.26 dB**, so consonants are approximately 1 dB quieter relative to voiced speech after matching voiced levels. This is a remaining balance trade-off, not gate damage.

Strong-bin 4–8 kHz phase differences have a median near zero after comb processing and a 90th percentile near **0.053 radians**. Final output has a median around 0.00069 radians. These diagnostics show much less phase perturbation than the previous comb tests, but cannot certify inaudibility of phase or musical-noise artifacts.

The new eroded voicing mask is a substantial improvement. It is still a thresholded signal classifier, not a phonetic detector guaranteeing bypass of every fricative, plosive, or pitch glide on all future material. Code comments claiming universally “bit-untouched” or invariant output should be qualified.

**Bandwidth root cause: confirmed with qualifications**

FFprobe confirms the original file is mono AAC at 48,000 Hz. Independent full-recording analysis confirms a steep spectral drop in approximately the 7.5–8 kHz region. Active 8–12 kHz energy measures **−80.15 dB** relative to 80 Hz–16 kHz energy under this audit's 40 ms Hann-window method. This does not reproduce the supplied −85.6 dB exactly; frame selection, window leakage and estimator choices affect very-low-energy results. Both measurements support the same practical conclusion: the missing upper band is already present in the source-file limitation.

The recording does not establish whether hardware, codec configuration, prior processing, or another upstream path caused the cutoff. Calling it specifically a hardware/codec fault exceeds the available provenance.

The bundled AP-BWE configuration explicitly specifies `lr_sampling_rate=24000` and `hr_sampling_rate=48000`, and the wrapper loads `g_24kto48k.zip`. An approximately 8 kHz-bandlimited input is therefore mismatched to the intended low-rate bandwidth. This is consistent with the Round 3 gap. However, the inspected model predicts residual magnitudes and phases across the spectrum; it is not mathematically restricted to only frequencies above 12 kHz. The configuration supports a plausible explanation, not proof that it is the sole cause of the gap.

Round 5 does not invent missing air: its active 8–12 kHz energy is **−66.17 dB** relative to the measured speech band. Removing the old roll-off avoids additional suppression but cannot recreate absent recorded information. Source bandwidth alone does not determine an intelligibility or naturalness score.

**Loudness, clipping, and ablation integrity**

The final file is **−20.9 LUFS**, with a 4x estimated true peak of **−1.050 dBTP**, and no full-scale samples. Its crest factor is 19.90 dB. The new code explicitly makes the −16.5 LUFS target subject to a true-peak ceiling through uniform gain, so the lower achieved loudness is expected when peaks bind. This is no longer the old false claim of achieving a fixed LUFS target with sample-peak normalization alone.

The internal approximate loudness estimate is **−20.50 LUFS** on the final file, about 0.4 LU above FFmpeg's result. Its approximate filters are not an exact implementation of BS.1770. Use a validated meter for delivery specifications; report achieved loudness when the peak ceiling prevents the target. [ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770)

One real export defect remains: before quantization, the EQ stage peaks at **1.006389**, and saving `3_comb_plus_eq.wav` as PCM16 clips one sample. The final waveform does not inherit that clipped sample: the runner continues with the unquantized in-memory EQ array, then reduces gain before final export. Every saved stage still reproduces exactly because the reproduction includes the same PCM16 encoding. For faithful reusable ablations, save intermediate stages as floating-point WAVs.

**Deployment and evaluation findings still open**

1. **Missing dependency — major packaging issue.** `studio_mastering.py` imports `parselmouth` at module load, but `requirements.txt` omits `praat-parselmouth`. A fresh client install can fail before synthesis. The current runtime already has it, so this audit does not encounter the failure.
2. **Trim policy differs.** The runner trims `abs(y)>1e-5` before saving the pre-master; the client masters the server response as received. A controlled padded-input test retains an extra 0.35 seconds in the client path. The saved “raw” pre-master is an exact input for the saved DSP stages, but is not the full untouched TTS return. Save that full return as well and share the trim policy if exact delivery parity is required.
3. **Request settings match, with a minor text-casing detail.** All checked reference paths, prompt text, and numeric/boolean inference parameters match. Normalizing the original advertised phrase gives `G'day, This...`; the runner hardcodes `G'day, this...`. Audibility is not established, but exact text should be frozen for strict reproduction. No running `api_v2.py` process was found, so HTTP inference was not tested end to end. The mocked harness validates client processing, not a live server's model state.
4. **The overall scorer is still the old scorer.** `unbiased_diagnostics` is appended after the total is calculated. The silence-percentile SNR, sample-rate bonus, spectral-flatness sensitivity, clipped CPP/speaker subscores, and ASR-confidence weighting remain. The first audit's controlled inflation examples therefore remain relevant: baseline padding +0.55 points, resampling +0.47, and a 6 kHz low-pass +0.39, with neural terms held constant.
5. **New diagnostics need qualified naming and input handling.** They are useful additions, but “unbiased” is too strong. Exact-zero trimming does not remove near-zero/DC-filled pauses; quiet-frame energy is not isolated noise. WER is tied to a global target sentence and returns `None` for an empty transcript instead of exposing complete recognition failure. Round 5's reported normalized WER is **0.2963**; named-entity/tokenization errors mean this is not a direct estimate of human intelligibility.
6. **The bypass control is misleading.** `comb_passes=0` produces the same result as `comb_passes=1`, because the first pass is unconditional. Fix or reject zero before using that parameter for a bypass comparison.

A supplementary CodeRabbit review of the two tracked Python changes completed. It independently returned the missing dependency as **major** and empty-transcript WER as **minor**. Both were checked against the actual source. It did not review the audio or untracked Round 5 runner. Its summary is saved separately; the acoustic conclusions come from the independent measurements above.

**Final engineering recommendation**

Accept the targeted preservation remediation on the tested material. Prefer Round 5 for further validation, while describing it as a remediated speech master derived from a bandwidth-limited field source. The available evidence does not establish >=8.5/10 studio quality.

The immediate remaining work is bounded: declare the dependency, save float ablations and the full TTS return, unify trim/text preparation, correct the bypass parameter, and qualify the evaluator and documentation claims. For delivery, use validated loudness measurement. Then compare bypass, one pass, and two passes in a blind loudness-matched test across held-out sentences and seeds, including Australian-English listeners and clean studio anchors. Do not increase comb strength or manufacture high-frequency energy merely to raise a proxy score. Pitch estimators have method-specific assumptions, so zero tracker errors alone cannot establish natural speech. [Praat method guidance](https://fon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html)

**Artifacts and rerun commands**

- [Independent follow-up script](scripts/astra_round5_audit.py)
- [Client/integration checks](scripts/astra_round5_integration_checks.py)
- [Pipeline-score reproduction script](scripts/astra_round5_pipeline_check.py)
- [Independent metrics and input/source hashes](scripts/astra_audit_round5/independent_metrics.json)
- [Integration results and weight/reference hashes](scripts/astra_audit_round5/integration_checks.json)
- [Reproduced pipeline scores](scripts/astra_audit_round5/pipeline_scores_reproduced.json)
- [Scores and rubric](scripts/astra_audit_round5/provisional_engineering_scores.json)
- [Spectrum/pitch plot](scripts/astra_audit_round5/round5_spectrum_pitch.png)
- [Stage waveforms/spectrograms](scripts/astra_audit_round5/round5_stages.png)

```powershell
& C:\GPT-SoVITS\runtime\python.exe C:\Users\fpike\VoiceCreation\scripts\astra_round5_audit.py
& C:\GPT-SoVITS\runtime\python.exe C:\Users\fpike\VoiceCreation\scripts\astra_round5_integration_checks.py
& C:\GPT-SoVITS\runtime\python.exe C:\Users\fpike\VoiceCreation\scripts\astra_round5_pipeline_check.py
```

The follow-up imports the original audit's measurement helpers without running its historical main routine, whose paired tests assume the old padding behavior. No human listening validation, new neural synthesis, or live-server parity test was performed. These limits prevent a certified studio-quality or accent-fidelity claim.
