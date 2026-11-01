**Round 4 shows genuine acoustic improvement, but I would not certify it as ≥8.5/10 studio quality.** Its published **8.60/10** is reproducible, yet the evaluator substantially rewards silence padding, resampling, and spectral shaping. My provisional engineering assessment is **7.00/10**.

I ran the independent analysis using `C:\GPT-SoVITS\runtime\python.exe`. **I did not conduct an auditory listening test.** The scores below are engineering judgments supported by measurements and code inspection—not listening-derived MOS, verified accent fidelity, or hundredth-point precision.

**Scope and evidence**

- Analyzed all five renders and all four reference WAVs.
- References exist under `ref\`; the requested `reference\` directory and `iter*_metrics.json` files were absent.
- Re-ran the original evaluator separately and reproduced every published overall score exactly.
- Verified that the deployed `samples\francisv2_sample.wav` is byte-identical to Round 4.
- Confirmed 75 training-list clips totaling **372.77 seconds—6.21 minutes**. These are conditioned field recordings; “studio clips” does not establish studio-recorded source quality.
- Left the production code, weights, references, and renders unchanged.

The independent analysis uses Praat pitch/harmonicity, separate spectral measurements, Burg formant estimates, FFmpeg loudness measurements, and controlled DSP ablations. Pitch and harmonicity measurements depend on analysis method and phonetic content; they are diagnostics rather than standalone quality scores. [Praat harmonicity documentation](https://uvafon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__ac____.html)

**Round-by-round scores**

My provisional rubric equally weights speech-band artifact control, useful spectral balance, pitch/envelope integrity, and level/dynamics/pause handling. It awards no points for sample rate alone or added silence. Differences below roughly one point should not be interpreted as established perceptual differences.

| Round | Independent provisional score | Pipeline score, reproduced |
|---|---:|---:|
| 0 — Baseline | **5.00 / 10.0** | 4.99 |
| 1 — Curation and conditioned references | **6.00 / 10.0** | 6.70 |
| 2 — Retrained v2ProPlus | **6.75 / 10.0** | 6.95 |
| 3 — Low-noise prior and AP-BWE | **6.50 / 10.0** | 6.84 |
| 4 — Studio Master | **7.00 / 10.0** | 8.60 |

The principal independent measurements are:

| Measurement | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---:|---:|---:|---:|---:|
| Duration, seconds | 11.50 | 10.08 | 8.62 | 11.06 | 10.36 |
| Median Praat HNR, dB¹ | 5.17 | 4.10 | 5.30 | 5.83 | 6.30 |
| Median F0, Hz² | 119.7 | 144.7 | 137.2 | 161.7 | 150.8 |
| Inter-harmonic/harmonic density, 0.5–4 kHz, dB³ | −6.96 | −9.25 | −11.14 | −12.70 | −16.99 |
| Inter-harmonic/harmonic density, 4–8 kHz, dB³ | −0.37 | −0.48 | −0.58 | −0.69 | −4.45 |
| Integrated loudness, LUFS | −20.7 | −22.5 | −21.5 | −20.4 | −19.5 |
| Active-speech crest factor, dB | 16.88 | 18.08 | 20.33 | 19.26 | 18.69 |
| Approximate 4× oversampled peak, dBTP | −3.34 | −4.47 | −1.38 | −1.11 | −1.08 |
| Exact-zero samples | 8.02% | 8.47% | 5.89% | 0.84% | 11.38% |

¹ Median over valid Praat analysis frames, not identical sustained vowels.  
² Confidently voiced frames; intonation changes are not automatically instability.  
³ Stationary, confidently voiced frames. More negative means deeper valleys between harmonics. This can reflect either useful noise suppression or artificially imposed periodicity; it is not a clean-reference noise measurement.

**Round 0 — Baseline: 5.00/10**

The baseline has the weakest measured mid-band harmonic separation and substantially less upper-midrange energy. Active 2–4 kHz energy is **−17.94 dB** relative to the measured 80 Hz–16 kHz energy, consistent with comparatively weak presence.

Its strengths are reasonable headroom, no full-scale clipping, and stable confidently tracked pitch: no adjacent pitch steps exceeded three semitones under the selected analysis settings.

The main limitations are weak presence and relatively unstructured voiced high-frequency energy. However, those measurements alone cannot distinguish all natural breathiness from vocoder fuzz. The file already contains considerable digital silence, so its pause floor is not an untreated acoustic noise reference.

**Round 1 — Curation and conditioned references: 6.00/10**

Presence improves substantially: active 2–4 kHz energy rises to **−13.67 dB**, and mid-band inter-harmonic separation improves from **−6.96 to −9.25 dB**.

That supports a useful improvement in spectral presentation. The reproduced speaker embedding similarity also rises from **0.7689 to 0.8333**, although that is supporting pipeline evidence rather than independent proof of timbre fidelity.

The improvement is less convincing for high-frequency de-fuzzing. The independent 4–8 kHz valley ratio barely changes, while median Praat HNR falls to **4.10 dB**. This does not prove worse sound—the utterance and intonation changed—but it contradicts treating all harmonic-quality indicators as uniformly improved.

Also, Round 1’s runner saves the synthesized output directly. The Wiener processing conditions the dataset/references; it is not a demonstrated post-synthesis Wiener cleanup of this render. Retraining occurs in Round 2.

**Round 2 — Retrained v2ProPlus: 6.75/10**

The training list and Stage 2 log support the claimed 75-clip retraining, including completion of ten Stage 2 epochs.

This render improves mid-band harmonic separation to **−11.14 dB** and maintains the stronger presence established in Round 1. Confident pitch tracking shows no adjacent jumps above three semitones.

It is the shortest render, **8.62 seconds**, and has the highest active-speech crest factor, **20.33 dB**. Those are reasons to check pacing and transient balance in listening, rather than evidence of defects by themselves.

Remaining limitations include nearly unchanged 4–8 kHz inter-harmonic structure and a steep spectral drop around 7–8 kHz. Its high ASR-confidence score does not establish correct articulation: the reproduced transcript still includes substitutions such as “Today” and “GPT-Soviet.” Recognizer errors require listening confirmation before assigning them to the synthesis.

**Round 3 — Low-noise prior and AP-BWE: 6.50/10**

Median Praat HNR improves to **5.83 dB**, and mid-band harmonic separation reaches **−12.70 dB**. Pitch tracking remains generally stable.

AP-BWE adds measurable high-frequency output, but its distribution is questionable as evidence of restored natural air. Active **8–12 kHz** energy remains approximately **−64.66 dB** relative to the measured speech-band energy, whereas **12–16 kHz** rises to **−42.88 dB**. The spectrum therefore shows a pronounced gap followed by renewed upper-frequency energy.

This is a spectral discontinuity worth investigating—not proof of audible aliasing or a particular phase defect. The 4–8 kHz valley ratio remains close to earlier rounds.

Round 3 also changes references, sampling parameters, pause spacing, and inference noise. Its differences cannot be attributed to `noise_scale=0.25` or AP-BWE individually. The speaker subscore reaches **10.0 through numerical saturation**, not demonstrated perfect identity.

**Round 4 — Studio Master: 7.00/10**

Round 4 delivers the clearest measurable reduction in voiced inter-harmonic energy:

- **0.5–4 kHz:** −16.99 dB.
- **4–8 kHz:** −4.45 dB, versus −0.69 dB in Round 3.
- Median Praat HNR: **6.30 dB**.
- No full-scale clipping; estimated true peak approximately **−1.08 dBTP**.

These are real improvements. They are not explained entirely by silence padding.

The trade-offs prevent a studio-quality certification. Useful spectral content still falls sharply around **7–8 kHz**. Round 4’s 8–12 kHz energy is approximately **−73.56 dB** relative to the measured speech-band energy: the 48 kHz container does not demonstrate restored high-frequency detail.

The strongest preservation concerns arise from controlled tests of the mastering chain, described below. Round 4 also has more large adjacent pitch-tracker steps—**1.52%**—although tracker ambiguity and changed harmonic weighting prevent interpreting that directly as audible pitch glitches.

Finally, Round 4 is a fresh synthesis using `noise_scale=0.45`, changed references, and changed sampling settings. It replaces AP-BWE with polyphase resampling. It is **not simply Round 3 passed through mastering**, and its exact pre-master waveform was not saved for comparison.

**Evaluator audit: demonstrated score inflation**

I tested unchanged speech with controlled transformations. Neural speaker and ASR terms were held constant in these experiments; the changes below are weighted overall-score contributions from the acoustic terms.

| Controlled change | Result |
|---|---|
| Add one second of silence at each end of the baseline | **+0.55 points**; reported “SNR” rises from 45.41 to 103.89 dB |
| Resample baseline from 32 to 48 kHz only | **+0.47 points**, without adding source information |
| Low-pass baseline at 6 kHz | **+0.39 points**; de-fuzz subscore rises from 4.61 to 7.91 |
| Remove Round 4’s outer exact-zero padding | **−0.53 points**; reported “SNR” falls from 104.31 to 41.58 dB |

These results expose several problems in [evaluate_quality.py](C:/Users/fpike/VoiceCreation/scripts/evaluate_quality.py:134):

1. **“SNR” is an energy-percentile spread.** It does not separate clean speech from noise. The numerical RMS floor and zero padding produce the advertised 104 dB result.

2. **The selected “voiced” frames are merely the loudest 45%.** This can include fricatives and exclude quieter vowels. Adding silence also changes which speech frames enter the analysis.

3. **Spectral flatness conflates spectral shape with noise.** Removing frequency bands lowers flatness. A flatness reduction of 93.2% is not evidence that 93.2% of vocoder noise was removed.

4. **Resampling earns an explicit bonus and changes the CPP proxy.** In the controlled resampling test, the CPP proxy rises from **0.560 to 0.607**. Its dependence on FFT/sample-rate choices adds inflation beyond the explicit sample-rate reward.

5. **Several subscores saturate early.** CPP reaches its maximum at approximately **0.709**, already exceeded by Round 2. The silence score saturates at **66 dB**; speaker fidelity saturates at cosine similarity **0.875**.

6. **ASR confidence is not transcription accuracy.** `TARGET_SENTENCE` is not used to calculate word or character errors. A confidently incorrect transcript can receive a high articulation score.

7. **The evaluator lacks preservation penalties.** It does not directly assess clipped consonants, artificial tonality, formant changes, pitch-tracker disruptions, true peak, or actual integrated loudness.

The scorer is useful as a collection of development diagnostics. Its weighted total is not a validated studio-quality scale.

**DSP audit: preservation claims exceed the evidence**

The principal issue in [studio_mastering.py](C:/Users/fpike/VoiceCreation/studio_mastering.py:39) is the voicing floor:

```python
voicing = np.clip((ac_norm - 0.15) / 0.35, 0.25, 1.0)
```

Every frame receives comb processing, including unvoiced consonants. There is also no temporal smoothing of the selected pitch period.

I applied the current chain to identical surviving inputs, comparing each processing stage against its input. These tests isolate DSP behavior, although they cannot reconstruct the missing Round 4 pre-master waveform.

| Paired-test finding after two comb passes | Round 2 input | Primary reference input |
|---|---:|---:|
| Previously unvoiced frames now confidently classified as voiced | 19.69% | 25.51% |
| Median unvoiced-frame gain | +1.33 dB | +1.30 dB |
| Median broad-envelope shape error | 0.54 dB | 0.54 dB |
| 90th-percentile absolute F3 tracker shift | 62.7 Hz | 127.4 Hz |

The envelope correction is doing useful work: broad-envelope changes are generally modest. Nevertheless, **“100% preserved” and “unvoiced consonants untouched” are unsupported**. Tracker changes can reflect altered harmonic weighting rather than literal changes in vocal-tract resonances or perceived pitch.

Additional findings:

- **Phase preservation is not exact after reconstruction.** Reanalysis of the two-pass output shows median absolute strong-bin phase differences around **0.13 radians** in 4–8 kHz, with 90th percentiles around **0.53 radians**. This does not establish audible phase damage, but retaining input STFT phase does not guarantee unchanged output phase.
- **The harmonic-contrast stage can sharpen random peaks.** Its frequency median comparison operates on unvoiced material too, creating a plausible mechanism for granular or tonal artifacts. Audible musical noise remains unverified.
- **The expander can affect weak consonants.** In the reference-input test, **13.27%** of active unvoiced frames received more than 6 dB attenuation after removing normalization gain. The Round 2-input result was only **0.62%**, showing strong dependence on the material.
- **The “limiter” is peak normalization.** It measures neither LUFS nor true peak. The actual Round 4 result is **−19.5 LUFS**, despite the −16 LUFS claim. Loudness and true-peak measurement are distinct operations. [ITU-R BS.1770](https://www.itu.int/rec/r-rec-bs.1770/_page.print)
- **The EQ documentation is stale.** It describes a high-frequency air shelf; the implementation applies a roll-off approaching **−12 dB**, centered around 8.2 kHz.

No full-scale clipping was detected in any render. That does not exclude nonlinear distortion introduced earlier by conditioning, and ordinary speech does not support a meaningful clean-reference THD measurement here.

**Reproducibility and deployment findings**

The training script deletes the previous experiment directory and overwrites legacy-named weights. Reference files are also overwritten between rounds. Consequently, the archived WAVs are stronger historical evidence than rerunning those scripts against today’s files.

There is also a deployment mismatch: [voice_agent.py](C:/Users/fpike/VoiceCreation/voice_agent.py:173) does not reproduce Round 4’s `noise_scale=0.45` override. The inspected decoder defaults to **0.5**. The deployed example matches Round 4 exactly, but future client synthesis is not guaranteed to reproduce its inference configuration.

**Recommended next changes, in priority order**

1. **Make the comparison controlled.** Save the exact pre-master waveform and freeze text, references, checkpoint hashes, seed, and inference settings. Compare bypass, one comb pass, two passes, EQ, and gate on that same waveform.

2. **Make unvoiced bypass real.** Allow zero comb strength on unvoiced frames, smooth voicing transitions and pitch estimates, and bypass uncertain or rapidly changing pitch. Start by testing one gentler pass.

3. **Disable harmonic contrast on fricatives.** Preserve natural noise-like consonants. Measure voiced and unvoiced spectra separately.

4. **Reduce the gate’s influence on speech.** Test a gentler expansion curve with hold/release behavior and inspect weak word endings. Exclude delivery padding from quality calculations.

5. **Treat the bandwidth limitation at its source.** Inspect raw versus conditioned training spectra and the synthesis path around the 7–8 kHz cutoff. Neither a 3.9 kHz boost nor resampling reconstructs missing detail. Keep AP-BWE only if a controlled comparison demonstrates benefit.

6. **Replace the overall self-score with separate diagnostics.** Remove the sample-rate bonus, report standardized harmonicity, evaluate transcript accuracy, and retain saturation-free raw measurements. Do not relabel silence-percentile spread as SNR.

7. **Run blind, loudness-matched listening validation.** Include clean Australian-English anchors, several held-out sentences, multiple seeds, and separate ratings for naturalness, identity, articulation, and noise. Subjective speech-quality methods explicitly depend on controlled listening evidence. [ITU-T P.800](https://www.itu.int/itu-t/recommendations/rec.aspx?lang=en&rec=3638)

**Verdict:** retain Round 4 as a promising processed candidate. Its improvement is measurable, but **the ≥8.5/10 studio-quality claim is not established**. Further progress should focus on preserving consonants and natural harmonic structure, followed by controlled listening—not increasing the existing score.

All audit artifacts are saved:

- [Independent verification script](C:/Users/fpike/VoiceCreation/scripts/astra_independent_audit.py)
- [Independent measurements, ablations, versions, and hashes](C:/Users/fpike/VoiceCreation/scripts/astra_audit/independent_metrics.json)
- [Reproduced pipeline scores](C:/Users/fpike/VoiceCreation/scripts/astra_audit/pipeline_scores_reproduced.json)
- [Provisional scoring rubric](C:/Users/fpike/VoiceCreation/scripts/astra_audit/provisional_engineering_scores.json)
- [Waveforms and spectrograms](C:/Users/fpike/VoiceCreation/scripts/astra_audit/waveforms_spectrograms.png)
- [Spectrum and pitch plots](C:/Users/fpike/VoiceCreation/scripts/astra_audit/spectrum_pitch.png)

The remaining uncertainty is perceptual: audible musical noise, accent fidelity, speaker identity, and studio-level naturalness require listening validation.