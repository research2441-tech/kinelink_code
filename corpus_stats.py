import glob, json, numpy as np, os
import gen_data as G
D='./data'
splits=['train','val','S0_walk_icu','S1_care_icu','S2_run_icu','S3_walk_ftpr','S4_walk_icu_crowded']
out={}
for sp in splits:
    fs=sorted(glob.glob(f'{D}/{sp}__*.npz'))
    clips=sorted(set(f.split('__')[1] for f in fs)); subj=sorted(set(c.split('_')[0] for c in clips))
    T=0; st={}
    for f in fs:
        d=np.load(f); T+=len(d['t'])
        for nd in ('shoulder','chest','wrist','ankle'):
            for l in ('dl','ul'):
                z=10*np.log10(d[f'{nd}_{l}_H0']); st.setdefault((nd,l),[[],[],[]])
                st[(nd,l)][0].append(z); st[(nd,l)][1].append(np.abs(np.diff(z))); st[(nd,l)][2].append(d[f'{nd}_{l}_tau']*1e9)
    s={}
    for (nd,l),(z,dz,tau) in st.items():
        z=np.concatenate(z);dz=np.concatenate(dz);tau=np.concatenate(tau)
        s[f'{nd}_{l}']=dict(mean=float(z.mean()),std=float(z.std()),p99dz=float(np.percentile(dz,99)),tau=float(np.median(tau)),tau_p90=float(np.percentile(tau,90)))
    out[sp]=dict(files=len(fs),clips=len(clips),subjects=subj,frames=T,minutes=T/100/60,stats=s)
    print(sp,len(fs),len(clips),subj,T, '%.1f min'%(T/6000))
json.dump(out,open('results/corpus_stats.json','w'),indent=1)
for sp in ['S0_walk_icu','S3_walk_ftpr']:
    for l in ('ul','dl'):
        print(sp,l,' '.join('%s %.1f±%.1f d%.2f tau%.1f'%(nd[:2],out[sp]['stats'][f'{nd}_{l}']['mean'],out[sp]['stats'][f'{nd}_{l}']['std'],out[sp]['stats'][f'{nd}_{l}']['p99dz'],out[sp]['stats'][f'{nd}_{l}']['tau']) for nd in ('shoulder','chest','wrist','ankle')))
