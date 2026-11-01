"""Independent acoustic audit. Run with C:\\GPT-SoVITS\\runtime\\python.exe.

No auditory/MOS claim: these are signal diagnostics, not a listening test.
The pipeline evaluator is used ONLY in the explicitly labelled bias experiments.
Production audio/code is never overwritten. All output goes to scripts/astra_audit/.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import numpy as np
import scipy
from scipy import signal, ndimage
import soundfile as sf
import parselmouth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scripts' / 'astra_audit'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import studio_mastering as mastering

def db(x):
    return 20 * np.log10(np.maximum(x, 1e-15))

def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(zip(['p10', 'median', 'p90'], map(float, np.percentile(x, [10, 50, 90])))) if len(x) else None

def frames(y, sr, seconds=.04, hop=.01):
    n, h = round(seconds * sr), round(hop * sr)
    return np.lib.stride_tricks.sliding_window_view(y, n)[::h], (np.arange((len(y)-n)//h+1)*h+n/2)/sr

def mono(path):
    y, sr = sf.read(path, always_2d=True)
    return y.mean(axis=1), sr

def resample(y, sr, target):
    g = math.gcd(sr, target)
    return signal.resample_poly(y, target//g, sr//g)

def runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def pitch_features(y, sr):
    snd = parselmouth.Sound(y, sr)
    pitch = snd.to_pitch_ac(time_step=.01, pitch_floor=65, pitch_ceiling=350,
                            very_accurate=True, voicing_threshold=.5)
    f0 = pitch.selected_array['frequency']
    strength = pitch.selected_array['strength']
    hnr = snd.to_harmonicity_ac(time_step=.01, minimum_pitch=65,
                              silence_threshold=.03, periods_per_window=4.5)
    hv = hnr.values[0]
    hv = hv[(hv > -100) & np.isfinite(hv)]
    good = (f0 > 0) & (strength >= .6)
    adjacent = good[1:] & good[:-1]
    steps = np.abs(12*np.log2(np.maximum(f0[1:], 1)/np.maximum(f0[:-1], 1)))[adjacent]
    return {'f0_hz': stats(f0[good]), 'voiced_frame_fraction': float(np.mean(good)),
            'hnr_praat_db': stats(hv), 'adjacent_f0_step_semitones': stats(steps),
            'adjacent_steps_over_3_semitones_pct': float(100*np.mean(steps > 3)) if len(steps) else None}, pitch

def formants(y, sr, times):
    # Burg estimates are diagnostics, not labelled-vowel ground truth.
    snd = parselmouth.Sound(resample(y, sr, 16000), 16000)
    fm = snd.to_formant_burg(time_step=.01, max_number_of_formants=5,
                            maximum_formant=5000, window_length=.025,
                            pre_emphasis_from=50)
    return np.array([[fm.get_value_at_time(k, float(t)) for k in [1, 2, 3]] for t in times])

def loudness(path):
    cmd = [r'C:\GPT-SoVITS\runtime\ffmpeg.exe', '-hide_banner', '-nostats', '-i', str(path),
           '-af', 'ebur128=peak=true', '-f', 'null', '-']
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        return {'error': p.stderr[-1000:]}
    summary = p.stderr.rsplit('Summary:', 1)[-1]
    found = re.search(r'I:\s*([-\d.]+) LUFS.*?LRA:\s*([-\d.]+) LU.*?Peak:\s*([-\d.]+) dBFS', summary, re.S)
    return dict(zip(['integrated_lufs', 'lra_lu_short_clip_caution', 'true_peak_dbtp_ffmpeg'], map(float, found.groups()))) if found else {'raw_summary': summary}

def analyze(path):
    y, sr = mono(path)
    f, times = frames(y, sr)
    rms = np.sqrt(np.mean(f*f, axis=1))
    rms_db = db(rms)
    threshold = float(np.percentile(rms_db, 95)-30)
    active = rms_db > threshold
    nz = np.flatnonzero(y != 0)
    interior = (times >= nz[0]/sr+.1) & (times <= nz[-1]/sr-.1)
    quiet = interior & ~active
    pf, pitch = pitch_features(y, sr)
    pidx = np.clip(np.round((times-pitch.x1)/pitch.dx).astype(int), 0, pitch.nx-1)
    f0 = pitch.selected_array['frequency'][pidx]
    strength = pitch.selected_array['strength'][pidx]
    voiced = active & (f0 > 0) & (strength > .6)
    unvoiced = active & ~voiced
    nfft = 2**int(np.ceil(np.log2(f.shape[1]*2)))
    z = np.fft.rfft(f*np.hanning(f.shape[1]), n=nfft, axis=1)
    p = np.abs(z)**2
    freqs = np.fft.rfftfreq(nfft, 1/sr)
    total = p[:, (freqs >= 80) & (freqs < 16000)].sum(axis=1)
    bands = {}
    for lo, hi in [(80, 1500), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 23000)]:
        mask = (freqs >= lo) & (freqs < min(hi, sr/2))
        if not mask.any():
            continue
        bp = p[:, mask]
        flat = np.exp(np.mean(np.log(bp+1e-20), axis=1))/(bp.mean(axis=1)+1e-20)
        bands[f'{lo}_{hi}'] = {label: {'energy_relative_80_16k_db': float(10*np.log10((bp[m].sum()+1e-20)/(total[m].sum()+1e-20))),
                                     'flatness_median': float(np.median(flat[m]))} if np.any(m) else None
                              for label, m in [('active', active), ('voiced', voiced), ('unvoiced', unvoiced)]}
    # Longer windows for harmonic/valley discrimination. High F0 slope frames excluded.
    hf, ht = frames(y, sr, .06)
    hi = np.clip(np.round((ht-pitch.x1)/pitch.dx).astype(int), 0, pitch.nx-1)
    hp = pitch.selected_array['frequency'][hi]
    hs = pitch.selected_array['strength'][hi]
    st = np.r_[False, np.abs(np.diff(hp)) < 2] & np.r_[np.abs(np.diff(hp)) < 2, False]
    good = (hp > 0) & (hs > .7) & st & (db(np.sqrt(np.mean(hf*hf, axis=1))) > threshold)
    nn = 2**int(np.ceil(np.log2(hf.shape[1]*4)))
    hz = np.fft.rfftfreq(nn, 1/sr)
    hspec = np.abs(np.fft.rfft(hf[good]*np.hanning(hf.shape[1]), n=nn, axis=1))**2
    valley = {}
    for lo, high in [(500, 4000), (4000, 8000)]:
        ratios = []
        for row, fund in zip(hspec, hp[good]):
            distance = np.abs(hz/fund-np.round(hz/fund))
            band = (hz >= lo) & (hz < high)
            harm = band & (distance <= .18)
            inter = band & (distance >= .32)
            ratios.append(10*np.log10((np.mean(row[inter])+1e-20)/(np.mean(row[harm])+1e-20)))
        valley[f'{lo}_{high}_valley_to_harmonic_density_db'] = stats(ratios)
    valley['eligible_stationary_frames'] = int(good.sum())
    # Phase second difference near harmonic ridges; NOT an audible phase-distortion score.
    phase = np.angle(z)
    acceleration = np.angle(np.exp(1j*(phase[2:]-2*phase[1:-1]+phase[:-2])))
    dist = np.abs(freqs[None,:]/np.maximum(f0[:,None],1)-np.round(freqs[None,:]/np.maximum(f0[:,None],1)))
    valid = voiced[2:] & voiced[1:-1] & voiced[:-2] & (np.abs(f0[2:]-f0[:-2]) < 3)
    ridge = (dist[1:-1] < .15) & (freqs[None,:] >= 4000) & (freqs[None,:] < 8000)
    strong = p[1:-1] > np.max(p[1:-1], axis=1, keepdims=True)*1e-4
    selected = acceleration[valid[:,None] & ridge & strong]
    fm = formants(y, sr, times[voiced])
    clip = np.sum(np.abs(y) >= 1-1/32768)
    zeros = runs(y == 0)
    result = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'sr': sr, 'duration_s': len(y)/sr, 'subtype': sf.info(path).subtype,
              'sample_peak_dbfs': float(db(np.max(np.abs(y)))),
              'oversampled_4x_peak_dbtp_estimate': float(db(np.max(np.abs(signal.resample_poly(y, 4, 1))))),
              'full_scale_samples': int(clip), 'dc_offset': float(y.mean()),
              'crest_whole_db': float(db(np.max(np.abs(y))/np.sqrt(np.mean(y*y)))),
              'crest_active_db': float(db(np.max(np.abs(y))/np.sqrt(np.mean(f[active]**2)))),
              'active_rms_dbfs': float(db(np.sqrt(np.mean(f[active]**2)))),
              'active_frame_rms_p90_p10_db': float(np.percentile(rms_db[active],90)-np.percentile(rms_db[active],10)),
              'activity_threshold_dbfs': threshold, 'interior_quiet_rms_dbfs': stats(rms_db[quiet]),
              'exact_zero_pct': float(np.mean(y == 0)*100),
              'leading_exact_zero_s': float(nz[0]/sr), 'trailing_exact_zero_s': float((len(y)-1-nz[-1])/sr),
              'zero_runs_over_20ms_s': [[a/sr,b/sr] for a,b in zeros if b-a >= .02*sr],
              'band_analysis': bands, 'interharmonic': valley, 'pitch': pf,
              'hf_ridge_phase_second_difference_abs_rad': stats(np.abs(selected)),
              'formant_burg_hz': {f'F{k+1}': stats(fm[:,k]) for k in range(3)},
              'ffmpeg_loudness': loudness(path)}
    return result, (y, sr, times, rms_db, freqs, p, pitch)

def pipeline_acoustic(y, sr):
    # Extract the ORIGINAL scoring function; stub neural terms to constants.
    tree = ast.parse((ROOT/'scripts/evaluate_quality.py').read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['clamp','evaluate_wav']]
    class Reader:
        @staticmethod
        def read(*a, **k): return y.astype(np.float32), sr
    ns = {'np': np, 'sf': Reader, 'os': __import__('os'),
          'get_speaker_centroid': lambda *a: np.array([1.]),
          'compute_sv_embedding': lambda *a: np.array([.8]),
          'evaluate_asr_clarity': lambda *a: {'avg_logprob': -.3, 'transcript': 'held constant'}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<original acoustic scorer>', 'exec'), ns)
    r = ns['evaluate_wav']('audit', 'controlled test; neural terms held constant')
    s = r['subscores_out_of_10']
    r['acoustic_contribution_out_of_7_5'] = .3*s['1_harmonic_clarity']+.25*s['2_defuzz_and_snr']+.2*s['3_studio_bandwidth_presence']
    del r['overall_quality_score']
    return r

def paired_test(y, sr):
    y = resample(y, sr, 48000)
    hp = signal.sosfiltfilt(signal.butter(2, 80, fs=48000, btype='highpass', output='sos'), y)
    c1 = mastering.formant_preserving_harmonic_comb(hp,48000)
    c2 = mastering.formant_preserving_harmonic_comb(c1,48000,alpha=.58,gamma=1.35)
    eq = mastering.studio_broadcast_eq_and_air(c2,48000)
    gated = mastering.apply_studio_noise_gate_and_limiter(eq,48000)[16800:-16800]
    f,t = frames(hp,48000)
    _,pitch = pitch_features(hp,48000)
    idx = np.clip(np.round((t-pitch.x1)/pitch.dx).astype(int),0,pitch.nx-1)
    f0 = pitch.selected_array['frequency'][idx]
    active = db(np.sqrt(np.mean(f*f,axis=1))) > np.percentile(db(np.sqrt(np.mean(f*f,axis=1))),95)-30
    voiced = active & (f0 > 0) & (pitch.selected_array['strength'][idx] > .7)
    unvoiced = active & (f0 == 0)
    nfft=4096
    z0=np.fft.rfft(f*np.hanning(f.shape[1]),n=nfft,axis=1)
    env0=ndimage.gaussian_filter1d(np.abs(z0),18,axis=1)
    fr=np.fft.rfftfreq(nfft,1/48000)
    mask=(fr>=200)&(fr<=5000)
    fm0=formants(hp,48000,t[voiced])
    result={}
    for name,x in [('comb_once',c1),('comb_twice',c2),('comb_plus_eq',eq),('full_without_added_padding',gated)]:
        fx,_=frames(x,48000)
        zx=np.fft.rfft(fx*np.hanning(fx.shape[1]),n=nfft,axis=1)
        envx=ndimage.gaussian_filter1d(np.abs(zx),18,axis=1)
        delta=db(envx+1e-12)-db(env0+1e-12)
        delta-=np.median(delta[:,mask],axis=1,keepdims=True)
        fm=formants(x,48000,t[voiced])
        ratio=db(np.sqrt(np.mean(fx*fx,axis=1))/(np.sqrt(np.mean(f*f,axis=1))+1e-15))
        _,px=pitch_features(x,48000)
        pi=np.clip(np.round((t-px.x1)/px.dx).astype(int),0,px.nx-1)
        newf0=px.selected_array['frequency'][pi]
        confident=px.selected_array['strength'][pi]>.6
        both=voiced & (newf0>0) & confident
        f0error=np.abs(12*np.log2(newf0[both]/f0[both]))
        strong=(np.abs(z0)>np.max(np.abs(z0),axis=1,keepdims=True)*.01)
        phase_mask=strong & (fr[None,:]>=4000) & (fr[None,:]<8000) & active[:,None]
        phase_shift=np.abs(np.angle(zx*np.conj(z0)))
        result[name]={'voiced_envelope_shape_rmse_db':stats(np.sqrt(np.mean(delta[voiced][:,mask]**2,axis=1))),
                      'formant_absolute_shift_hz':{f'F{k+1}':stats(np.abs(fm[:,k]-fm0[:,k])) for k in range(3)},
                      'unvoiced_gain_db':stats(ratio[unvoiced]),
                      'original_unvoiced_now_confidently_voiced_pct':float(100*np.mean((newf0[unvoiced]>0)&confident[unvoiced])),
                      'matched_pitch_absolute_shift_semitones':stats(f0error),
                      'matched_pitch_shift_over_3_semitones_pct':float(100*np.mean(f0error>3)),
                      'strong_bin_phase_shift_4_8k_abs_rad':stats(phase_shift[phase_mask]),
                      'acoustic_pipeline':pipeline_acoustic(x,48000)}
    feq,_=frames(eq,48000);fg,_=frames(gated,48000)
    gate_ratio=db(np.sqrt(np.mean(fg*fg,axis=1))/(np.sqrt(np.mean(feq*feq,axis=1))+1e-15))
    gate_ratio-=np.median(gate_ratio[voiced])
    result['gate_only_normalization_removed']={'unvoiced_gain_db':stats(gate_ratio[unvoiced]),
                                              'unvoiced_frames_attenuated_over_6db_pct':float(100*np.mean(gate_ratio[unvoiced]<-6))}
    return result

def self_checks():
    sr=32000
    t=np.arange(sr*2)/sr
    tone=.1*np.sin(2*np.pi*140*t)
    p,_=pitch_features(tone,sr)
    assert abs(p['f0_hz']['median']-140)<1
    assert abs(float(db(np.max(np.abs(tone))/np.sqrt(np.mean(tone*tone))))-3.0103)<.01
    assert len(resample(tone,sr,48000))==96000
    white=np.random.default_rng(42).normal(0,.05,len(tone))
    pn,_=pitch_features(white,sr)
    assert pn['voiced_frame_fraction'] < .1
    return {'140_hz_pitch_recovery':True,'sine_crest_factor':True,'resample_length':True,'white_noise_unvoiced':True}

def main():
    results={'method': 'Praat pitch AC and harmonicity AC; independent NumPy/SciPy spectra; FFmpeg ebur128',
             'measurement_notes': {
                 'frames': '40 ms Hann, 10 ms hop; active RMS > file p95 minus 30 dB; voiced pitch strength > 0.6',
                 'pitch': 'Praat AC 65-350 Hz, 10 ms step; adjacent jumps are not clinical jitter',
                 'hnr': 'Praat AC, 4.5 periods/window at 65 Hz floor; all valid non-silent estimates, not only vowels',
                 'interharmonic': '60 ms Hann; strength > 0.7; adjacent F0 changes < 2 Hz; harmonic distance <= 0.18 F0 versus valleys >= 0.32 F0; mean power density ratio',
                 'formants': 'Burg, 16 kHz, 5 formants below 5 kHz; tracker shifts can reflect changed harmonic weighting',
                 'zero_floor': '-300 dB is a numerical log floor for zero RMS, not physical noise measurement',
                 'bias_experiments': 'Original acoustic formula only, speaker and ASR terms held constant; changes are weighted overall-point equivalents',
                 'paired_tests': 'Same input before/after current DSP, not Round 4 pre/post; broad envelopes measured after re-STFT',
                 'phase': 'Phase statistics do not distinguish natural phase variation from audible artifacts without a matched clean recording'},
             'limitations':['No listening test; no perceptual MOS certification.',
                            'Cross-round utterances are not sample-aligned. Formant distributions are not preservation tests.',
                            'Interharmonic and phase statistics are proxies, dependent on pitch tracking and phonetic content.',
                            'Current references are conditioned field recordings, not studio ground truth.',
                            'Paired DSP tests use surviving inputs, not the missing Round 4 pre-master waveform.'],
             'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'parselmouth':parselmouth.__version__},
             'self_checks':self_checks(),'files':{}}
    paths=[ROOT/'samples/iterations'/('iter0_baseline.wav' if i==0 else f'iter{i}_sample.wav') for i in range(5)]
    paths+=list(sorted((ROOT/'ref').glob('*.wav')))
    data={}
    for path in paths:
        print('Analyzing',path.name,flush=True)
        r,d=analyze(path)
        results['files'][path.stem]=r
        data[path.stem]=d
    results['deployed_sample_sha256']=hashlib.sha256((ROOT/'samples/francisv2_sample.wav').read_bytes()).hexdigest()
    results['deployed_matches_round4']=results['deployed_sample_sha256']==results['files']['iter4_sample']['sha256']
    # Common display scales; RMS normalize only these plots, not source samples.
    fig,axes=plt.subplots(5,2,figsize=(16,16))
    for i,key in enumerate(list(data)[:5]):
        y,sr,t,rms,f,p,pitch=data[key]
        axes[i,0].plot(np.arange(len(y))/sr,y,lw=.3)
        axes[i,0].set(title=key+' waveform',ylim=(-1,1),ylabel='Amplitude',xlabel='Seconds')
        norm=p/(np.mean(y*y)+1e-20)/(round(.04*sr)**2)
        axes[i,1].pcolormesh(t,f/1000,10*np.log10(norm.T+1e-15),vmin=-80,vmax=-10,shading='auto',rasterized=True)
        axes[i,1].set(title=key+' RMS-normalized spectrogram',ylim=(0,16),ylabel='kHz',xlabel='Seconds')
    fig.tight_layout();fig.savefig(OUT/'waveforms_spectrograms.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,9))
    for key in list(data)[:5]:
        y,sr,t,rms,f,p,pitch=data[key]
        active=rms>np.percentile(rms,95)-30
        ps=p[active].mean(axis=0)
        ps/=ps[(f>=80)&(f<16000)].sum()
        axes[0].plot(f,10*np.log10(ps/(f[1]-f[0])+1e-20),label=key)
        ff=pitch.selected_array['frequency'].copy();ff[ff==0]=np.nan
        axes[1].plot(pitch.xs(),ff,label=key,alpha=.7)
    axes[0].set(xlim=(0,16000),ylim=(-130,-20),xlabel='Hz',ylabel='Normalized PSD dB/Hz');axes[0].legend()
    axes[1].set(xlabel='Seconds',ylabel='F0 Hz');axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/'spectrum_pitch.png',dpi=150);plt.close(fig)
    baseline,sr=mono(paths[0]);r4,sr4=mono(paths[4])
    nz=np.flatnonzero(r4 != 0)
    results['metric_bias_experiments']={
        'baseline':pipeline_acoustic(baseline,sr),
        'baseline_plus_1s_silence_each_end':pipeline_acoustic(np.pad(baseline,(sr,sr)),sr),
        'baseline_resampled_48k_only':pipeline_acoustic(resample(baseline,sr,48000),48000),
        'round4_original':pipeline_acoustic(r4,sr4),
        'round4_outer_exact_zeros_removed':pipeline_acoustic(r4[nz[0]:nz[-1]+1],sr4),
        'baseline_lowpass_6k':pipeline_acoustic(signal.sosfiltfilt(signal.butter(6,6000,fs=sr,output='sos'),baseline),sr)}
    results['paired_dsp_tests']={}
    for path in [paths[2],ROOT/'ref/ref.wav']:
        print('Paired DSP test',path.name,flush=True)
        y,sr=mono(path);results['paired_dsp_tests'][path.stem]=paired_test(y,sr)
    rng=np.random.default_rng(42)
    hiss=signal.sosfiltfilt(signal.butter(4,[2000,7000],btype='bandpass',fs=48000,output='sos'),rng.normal(0,.05,96000))
    h1=mastering.formant_preserving_harmonic_comb(hiss,48000)
    h2=mastering.formant_preserving_harmonic_comb(h1,48000,alpha=.58,gamma=1.35)
    results['unvoiced_noise_comb_test']={label:pitch_features(v,48000)[0] for label,v in [('input_bandpass_noise',hiss),('two_pass_output',h2)]}
    # Audit files and source hashes pin this report to the exact reviewed versions.
    sources=['studio_mastering.py','voice_agent.py','scripts/evaluate_quality.py',
             'scripts/iter1_curate_and_condition.py','scripts/iter2_train_v2proplus.py',
             'scripts/run_iter3_defuzz_and_bwe.py','scripts/run_iter4_studio_master.py']
    results['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    listing=Path(r'C:\GPT-SoVITS\output\asr_opt\francisv2_studio.list')
    clips=[line.split('|')[0] for line in listing.read_text(encoding='utf-8').splitlines() if line.strip()]
    results['training_list']={'path':str(listing),'clips':len(clips),'total_seconds':sum(sf.info(p).duration for p in clips)}
    (OUT/'independent_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print('Wrote',OUT/'independent_metrics.json',flush=True)

if __name__=='__main__': main()
