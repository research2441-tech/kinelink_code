"""Aggregate link-adaptation results: seed means/SD, per-clip paired tests, violation bursts."""
import glob, pickle, os, json, itertools
import numpy as np
from scipy.stats import wilcoxon

RES = os.environ.get('RES_DIR', './results')
NODES = ('shoulder', 'chest', 'wrist', 'ankle')
SC = ['S0_walk_icu', 'S1_care_icu', 'S2_run_icu', 'S3_walk_ftpr', 'S4_walk_icu_crowded']
SC_NAME = {'S0_walk_icu': 'S0 walk/ICU', 'S1_care_icu': 'S1 care/ICU', 'S2_run_icu': 'S2 jog/ICU',
           'S3_walk_ftpr': 'S3 walk/FTPR', 'S4_walk_icu_crowded': 'S4 walk/ICU-6'}


def load(split, link, h, a=0.01, e=0.05):
    out = {}
    for f in sorted(glob.glob(f'{RES}/la_{split}_{link}_h{h}_a{a}_e{e}_s*.pkl')):
        s = int(f.split('_s')[-1][:-4])
        out[s] = pickle.load(open(f, 'rb'))
    return out


def bursts(v):
    v = np.nan_to_num(v, nan=0.0) > 0.5
    runs, n = [], 0
    for x in v:
        if x:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return runs


def metrics(runs, policy, nodes=NODES):
    """Per seed: SE, violation rate (any), mean-SER violation, mean burst length, per (node,clip) SE."""
    rows = []
    for s, d in runs.items():
        o = d['out']
        se, va, vm, b, pc = [], [], [], [], {}
        for nd in nodes:
            x = o[(nd, policy)]
            m = ~np.isnan(x['se'])
            se.append(x['se'][m]); va.append(x['va'][m]); vm.append(x['vm'][m])
            b += bursts(x['va'][m])
            clip = x['clip'][m]
            for c in np.unique(clip):
                pc[(nd, d['clips'][c])] = (float(np.mean(x['se'][m][clip == c])), float(np.mean(x['va'][m][clip == c])))
        rows.append(dict(se=float(np.mean(np.concatenate(se))), viol=float(np.mean(np.concatenate(va))),
                         viol_mean=float(np.mean(np.concatenate(vm))), burst=float(np.mean(b)) if b else 0.0,
                         burst_max=int(max(b)) if b else 0, per_clip=pc))
    return rows


def summary(runs, policy, nodes=NODES):
    r = metrics(runs, policy, nodes)
    f = lambda k: (float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r])))
    return dict(se=f('se'), viol=f('viol'), viol_mean=f('viol_mean'), burst=f('burst'),
                burst_max=max(x['burst_max'] for x in r), n_seeds=len(r))


def paired(runs, p1, p2, nodes=NODES):
    """Wilcoxon signed-rank on (node, clip) SE averaged over seeds; returns median diff, p, effect r."""
    a, b = metrics(runs, p1, nodes), metrics(runs, p2, nodes)
    keys = sorted(set.intersection(*[set(x['per_clip']) for x in a + b]))
    xa = np.array([np.mean([x['per_clip'][k][0] for x in a]) for k in keys])
    xb = np.array([np.mean([x['per_clip'][k][0] for x in b]) for k in keys])
    d = xa - xb
    if np.allclose(d, 0):
        return dict(n=len(d), mean_diff=0.0, p=1.0, r=0.0, wins=0)
    st = wilcoxon(xa, xb, zero_method='wilcox', alternative='two-sided', method='approx')
    z = abs(st.zstatistic) if hasattr(st, 'zstatistic') and st.zstatistic is not None else np.nan
    return dict(n=len(d), mean_diff=float(np.mean(d)), rel=float(np.mean(d) / max(np.mean(xb), 1e-9)),
                p=float(st.pvalue), r=float(z / np.sqrt(len(d))) if z == z else np.nan, wins=int((d > 0).sum()))


def holm(ps):
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.zeros(m)
    run = 0
    for i, j in enumerate(idx):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj
