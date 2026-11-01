"""Round 5 independent follow-up. Uses original audit measurements, never its old main().
Run with C:\\GPT-SoVITS\\runtime\\python.exe. Outputs are separate from the first audit.
No TTS regeneration, network calls, or changes to production files.
"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import numpy as np
import soundfile as sf
import parselmouth
from scipy import signal, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import astra_independent_audit as audit
import studio_mastering as dsp

OUT = ROOT / 'scripts/astra_audit_round5'
OUT.mkdir(exist_ok=True)
SR = 48000

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def pitch_at(y, times, method='ac'):
    snd = parselmouth.Sound(y, SR)
    p = getattr(snd, 'to_pitch_' + method)(time_step=.01, pitch_floor=65,
                pitch_ceiling=350, very_accurate=True, voicing_threshold=.5)
    idx = np.clip(np.round((times-p.x1)/p.dx).astype(int), 0, p.nx-1)
    return p.selected_array['frequency'][idx], p.selected_array['strength'][idx]

def compare(base, after):
    assert base.shape == after.shape
    f,t = audit.frames(base, SR)
    g,_ = audit.frames(after, SR)
    rb = np.sqrt(np.mean(f*f, axis=1)); ra = np.sqrt(np.mean(g*g, axis=1))
    active = audit.db(rb) > np.percentile(audit.db(rb),95)-30
    out = {}
    for method in ['ac','cc']:
        f0,st = pitch_at(base,t,method)
        f1,st1 = pitch_at(after,t,method)
        uv = active & (f0 == 0)
        voiced = active & (f0 > 0) & (st > .7)
        both = voiced & (f1 > 0) & (st1 > .6)
        delta = np.abs(12*np.log2(f1[both]/f0[both]))
        flips = (f1 > 0) & (st1 > .6) & uv
        gain = audit.db(ra/(rb+1e-15))
        # Remove only global level change. This is not a signal/noise ratio.
        gain -= np.median(gain[voiced])
        out[method] = {
            'active_unvoiced_frame_count':int(uv.sum()),
            'original_unvoiced_now_confidently_voiced_count':int(flips.sum()),
            'original_unvoiced_now_confidently_voiced_pct':float(100*flips.sum()/uv.sum()),
            'flip_times_seconds':t[flips].tolist(),
            'matched_voiced_frame_count':int(both.sum()),
            'matched_pitch_shift_over_3_semitones_pct':float(100*np.mean(delta>3)),
            'matched_pitch_absolute_shift_semitones':audit.stats(delta),
            'original_confident_voiced_lost_pct':float(100*np.mean(~((f1[voiced]>0)&(st1[voiced]>.6)))),
            'unvoiced_gain_global_level_removed_db':audit.stats(gain[uv]),
            'unvoiced_frames_attenuated_over_6db_pct':float(100*np.mean(gain[uv]<-6))}
        if method == 'ac':
            fv = voiced.copy()
    fft = 4096
    z0 = np.fft.rfft(f*np.hanning(f.shape[1]),n=fft,axis=1)
    z1 = np.fft.rfft(g*np.hanning(g.shape[1]),n=fft,axis=1)
    hz = np.fft.rfftfreq(fft,1/SR)
    mask = (hz>=200)&(hz<=5000)
    env0 = ndimage.gaussian_filter1d(np.abs(z0),18,axis=1)
    env1 = ndimage.gaussian_filter1d(np.abs(z1),18,axis=1)
    d = audit.db(env1+1e-12)-audit.db(env0+1e-12)
    d -= np.median(d[:,mask],axis=1,keepdims=True)
    out['voiced_envelope_shape_rmse_db'] = audit.stats(np.sqrt(np.mean(d[fv][:,mask]**2,axis=1)))
    fm0=audit.formants(base,SR,t[fv]); fm1=audit.formants(after,SR,t[fv])
    out['formant_absolute_shift_hz']={f'F{k+1}':audit.stats(np.abs(fm1[:,k]-fm0[:,k])) for k in range(3)}
    strong=np.abs(z0)>np.max(np.abs(z0),axis=1,keepdims=True)*.01
    pm=strong & active[:,None] & (hz[None,:]>=4000)&(hz[None,:]<8000)
    out['strong_bin_phase_shift_4_8k_abs_rad']=audit.stats(np.abs(np.angle(z1*np.conj(z0)))[pm])
    return out

def quantize(y, sr=SR):
    f=io.BytesIO()
    sf.write(f, y.astype(np.float32), sr, format='WAV', subtype='PCM_16')
    f.seek(0)
    return sf.read(f)[0]

def regenerate(y,sr):
    up=audit.resample(y,sr,SR)
    hp=signal.sosfiltfilt(signal.butter(2,80,fs=SR,btype='highpass',output='sos'),up)
    one=dsp.formant_preserving_harmonic_comb(hp,SR,alpha=.65,gamma=1.4)
    two=dsp.formant_preserving_harmonic_comb(one,SR,alpha=.5,gamma=1.25)
    eq=dsp.studio_broadcast_eq_and_air(two,SR)
    final=dsp.apply_studio_noise_gate_and_limiter(eq,SR)
    return [hp,one,two,eq,final]

def field_spectrum():
    path=Path(r'C:\Users\fpike\.gemini\antigravity\scratch\francis_voice_agent\voice_data\Glenroi_Av_2.m4a')
    cmd=[r'C:\GPT-SoVITS\runtime\ffmpeg.exe','-v','error','-i',str(path),'-ac','1','-ar','48000','-f','f32le','pipe:1']
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    n=1920; nfft=4096; window=np.hanning(n)
    hz=np.fft.rfftfreq(nfft,1/SR)
    bands=[(80,8000),(80,16000),(6000,7400),(8000,12000),(12000,16000)]
    rms_all=[]; band_all=[]; sums=np.zeros(len(hz));count=0
    # Read in small blocks to avoid loading 28 minutes of decoded audio into memory.
    while True:
        raw=p.stdout.read(n*250*4)
        if not raw: break
        a=np.frombuffer(raw,dtype='<f4').astype(np.float64)
        a=a[:len(a)//n*n].reshape(-1,n)
        if not len(a): break
        powers=np.abs(np.fft.rfft(a*window,n=nfft,axis=1))**2
        rms_all.extend(np.sqrt(np.mean(a*a,axis=1)))
        band_all.extend(np.stack([powers[:,(hz>=lo)&(hz<hi)].sum(axis=1) for lo,hi in bands],axis=1))
        sums+=powers.sum(axis=0);count+=len(a)
    error=p.stderr.read().decode(errors='replace');code=p.wait()
    assert code==0,error
    rms=np.array(rms_all); bp=np.array(band_all)
    active=audit.db(rms)>np.percentile(audit.db(rms),95)-30
    result={'path':str(path),'sha256':sha(path),'decoder':'FFmpeg float32, mono 48 kHz',
            'analyzed_seconds':count*n/SR,'active_frame_count':int(active.sum()),
            'active_threshold_dbfs':float(np.percentile(audit.db(rms),95)-30),
            'band_energy_relative_80_16k_db':{f'{lo}_{hi}':float(10*np.log10(bp[active,i].sum()/bp[active,1].sum())) for i,(lo,hi) in enumerate(bands)}}
    return result,hz,sums/count

def main():
    old=json.loads((ROOT/'scripts/astra_audit/independent_metrics.json').read_text())
    results={'limitations':['No listening test or perceptual MOS certification.',
                'Praat AC matches the previous audit; cross-correlation provides a second detector, not human ground truth.',
                'Saved pre-master has already been trimmed by the Round 5 runner.',
                'Speech-band cutoff can be established in the source; its hardware/codec cause cannot be determined from this file alone.'],
             'self_checks':audit.self_checks(),'files':{},'historical_hash_checks':{},
             'analysis_script_sha256':sha(__file__),'base_analysis_script_sha256':sha(ROOT/'scripts/astra_independent_audit.py')}
    it=ROOT/'samples/iterations'
    rounds=[it/('iter0_baseline.wav' if i==0 else f'iter{i}_sample.wav') for i in range(6)]
    stages=sorted((it/'ablations').glob('*.wav'))+[it/'iter5_sample.wav']
    paths=rounds+[it/'iter5_premaster_raw.wav']+stages[:-1]
    data={}
    for path in paths:
        print('Independent measurement:',path.name,flush=True)
        m,d=audit.analyze(path)
        results['files'][path.stem]=m;data[path.stem]=d
        if path.stem in old['files']:
            results['historical_hash_checks'][path.stem]=(m['sha256']==old['files'][path.stem]['sha256'])
    stage_y=[audit.mono(p)[0] for p in stages]
    results['saved_stage_preservation']={}
    for p,y in zip(stages[1:],stage_y[1:]):
        print('Paired saved stage:',p.name,flush=True)
        results['saved_stage_preservation'][p.stem]=compare(stage_y[0],y)
    results['gate_only_preservation']=compare(stage_y[3],stage_y[4])
    raw,sr=audit.mono(it/'iter5_premaster_raw.wav')
    regenerated=regenerate(raw,sr)
    results['stage_regeneration']={}
    for path,saved,rebuilt in zip(stages,stage_y,regenerated):
        q=quantize(rebuilt)
        results['stage_regeneration'][path.stem]={
            'same_length':len(saved)==len(q),'max_difference_pcm16_lsb':float(np.max(np.abs(saved-q))*32768),
            'identical_samples_pct':float(100*np.mean(saved==q))}
    master,_=dsp.master_synthesized_voice(raw,orig_sr=sr,target_sr=SR)
    results['public_master_function_vs_saved_final']={'max_difference_pcm16_lsb':float(np.max(np.abs(quantize(master)-stage_y[-1]))*32768)}
    results['cross_input_preservation']={}
    for path in [it/'iter2_sample.wav',ROOT/'ref/ref.wav']:
        y,sr=audit.mono(path); v=regenerate(y,sr)
        results['cross_input_preservation'][path.stem]={'two_comb_passes':compare(v[0],v[2]),'gate_only':compare(v[3],v[4])}
    print('Checking full original field recording spectrum',flush=True)
    result,hz,psd=field_spectrum();results['source_recording']=result
    results['deployed_matches_round5']=sha(ROOT/'samples/francisv2_sample.wav')==sha(it/'iter5_sample.wav')
    sources=['studio_mastering.py','voice_agent.py','requirements.txt','scripts/evaluate_quality.py','scripts/run_iter5_astra_remediated.py']
    results['source_hashes']={p:sha(ROOT/p) for p in sources}
    results['external_source_hashes']={str(p):sha(p) for p in [Path(r'C:\GPT-SoVITS\api_v2.py'),Path(r'C:\GPT-SoVITS\GPT_SoVITS\module\models.py'),Path(r'C:\GPT-SoVITS\GPT_SoVITS\TTS_infer_pack\TTS.py'),Path(r'C:\GPT-SoVITS\tools\audio_sr.py')]}
    results['approximate_vs_ffmpeg_loudness']={p.stem:{'dsp_estimate_lufs':dsp._estimate_k_weighted_lufs(y,SR),'ffmpeg':results['files'][p.stem]['ffmpeg_loudness']} for p,y in zip(stages,stage_y)}
    # Same input through old vs new DSP; load trusted locally committed implementation, no checkout.
    old_src=subprocess.run(['git','show','HEAD:studio_mastering.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    (OUT/'round4_mastering_snapshot.py').write_text(old_src,encoding='utf-8')
    oldns={};exec(compile(old_src,'round4_mastering_snapshot.py','exec'),oldns)
    old1=oldns['formant_preserving_harmonic_comb'](regenerated[0],SR,alpha=.76,gamma=1.65)
    old2=oldns['formant_preserving_harmonic_comb'](old1,SR,alpha=.58,gamma=1.35)
    results['old_comb_on_round5_input']=compare(regenerated[0],old2)
    # Fixed-level speech-band PSD avoids inferring quality from file sample rate.
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    psd/=psd[(hz>=80)&(hz<16000)].sum()
    axes[0].plot(hz,10*np.log10(psd/(hz[1]-hz[0])+1e-20),label='Original field recording, all frames',alpha=.7)
    for key in ['iter3_sample','iter4_sample','iter5_premaster_raw','iter5_sample']:
        y,sr,t,rms,f,p,pitch=data[key]
        active=rms>np.percentile(rms,95)-30
        avg=p[active].mean(axis=0);avg/=avg[(f>=80)&(f<16000)].sum()
        axes[0].plot(f,10*np.log10(avg/(f[1]-f[0])+1e-20),label=key)
    axes[0].set(xlim=(0,16000),ylim=(-140,-20),xlabel='Hz',ylabel='Normalized density dB/Hz');axes[0].legend()
    for p in stages:
        pitch=data[p.stem][-1]; f0=pitch.selected_array['frequency'].copy();f0[f0==0]=np.nan
        axes[1].plot(pitch.xs(),f0,label=p.stem,alpha=.7)
    axes[1].set(xlabel='Seconds',ylabel='Praat F0 Hz');axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/'round5_spectrum_pitch.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(len(stages),2,figsize=(14,14))
    for i,path in enumerate(stages):
        y,sr,t,rms,f,p,pitch=data[path.stem]
        axes[i,0].plot(np.arange(len(y))/sr,y,lw=.3)
        axes[i,0].set(title=path.stem,ylim=(-1,1),xlabel='Seconds')
        normalized=p/(np.mean(y*y)+1e-20)/(round(.04*sr)**2)
        axes[i,1].pcolormesh(t,f/1000,10*np.log10(normalized.T+1e-15),vmin=-80,vmax=-10,shading='auto',rasterized=True)
        axes[i,1].set(ylim=(0,10),xlabel='Seconds',ylabel='kHz')
    fig.tight_layout();fig.savefig(OUT/'round5_stages.png',dpi=120);plt.close(fig)
    (OUT/'independent_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved',OUT/'independent_metrics.json',flush=True)

if __name__=='__main__':main()
