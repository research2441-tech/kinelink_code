"""Build the results dictionary R (tables, statistics) and the result figures for the manuscript."""
import os, json, pickle, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze as A
import features as F

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5, 'axes.linewidth': 0.6,
                     'legend.frameon': False, 'mathtext.fontset': 'dejavuserif', 'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5})
RES = './results'
FIG = './figs'
MAIN = ['Oracle', 'KINELINK', 'GRU+ACI', 'LRU-IMU+ACI', 'AR+ACI', 'Persist+ACI', 'OLLA', 'LSTM+split', 'Q-learning', 'Fixed']
ABL = ['KINELINK', 'KL uniform margin', 'KL w/o IMU', 'KL fixed CP', 'KL coverage score', 'KL gain score', 'KL split CP']
SCEN = A.SC


def val_losses():
    out = {}
    for k in ('lstm_imu', 'lstm', 'gru', 'lru_imu', 'lru'):
        v = [min(h[1] for h in pickle.load(open(f'./models_out/{k}_s{s}.pkl', 'rb'))['hist']) for s in (0, 1, 2)]
        out[k] = (float(np.mean(v)), float(np.std(v)))
    return out


def forecast_table():
    r = pickle.load(open(f'{RES}/forecast_rmse.pkl', 'rb'))
    K = ['persist', 'ar', 'lstm', 'gru', 'lru', 'lru_imu', 'lstm_imu']
    out = {}
    for split in SCEN:
        for l in ('ul', 'dl'):
            for h in (1, 2, 5):
                for k in K:
                    vz = [np.mean([r[(split, k, sd, l, nd, h)][0] for nd in F.NODES]) for sd in (0, 1, 2) if (split, k, sd, l, 'shoulder', h) in r]
                    vt = [np.mean([r[(split, k, sd, l, nd, h)][1] for nd in F.NODES]) for sd in (0, 1, 2) if (split, k, sd, l, 'shoulder', h) in r]
                    out[(split, l, h, k)] = (float(np.mean(vz)), float(np.std(vz)), float(np.mean(vt)), float(np.std(vt)))
                # per node for lstm_imu vs persist
                for nd in F.NODES:
                    for k in ('persist', 'lstm', 'lstm_imu'):
                        v = [r[(split, k, sd, l, nd, h)][0] for sd in (0, 1, 2) if (split, k, sd, l, nd, h) in r]
                        out[(split, l, h, k, nd)] = float(np.mean(v))
    return out


def la_tables(h=5):
    out = {}
    for sc in SCEN:
        for l in ('ul', 'dl'):
            runs = A.load(sc, l, h)
            if not runs:
                continue
            for p in set(MAIN + ABL):
                out[(sc, l, p)] = A.summary(runs, p)
            # per-node SE for KINELINK and OLLA
            for p in ('KINELINK', 'OLLA', 'KL w/o IMU', 'Oracle'):
                for nd in F.NODES:
                    out[(sc, l, p, nd)] = A.summary(runs, p, nodes=(nd,))['se'][0]
    return out


def stats_table(h=5):
    out = {}
    comps = ['OLLA', 'Persist+ACI', 'AR+ACI', 'GRU+ACI', 'LRU-IMU+ACI', 'KL w/o IMU', 'KL fixed CP', 'KL coverage score', 'KL split CP', 'LSTM+split', 'Q-learning']
    for sc in SCEN:
        if sc == 'S2_run_icu':
            continue
        for l in ('ul', 'dl'):
            runs = A.load(sc, l, h)
            if not runs:
                continue
            res = {c: A.paired(runs, 'KINELINK', c) for c in comps}
            adj = A.holm(np.array([res[c]['p'] for c in comps]))
            for c, pa in zip(comps, adj):
                res[c]['p_holm'] = float(pa)
                out[(sc, l, c)] = res[c]
    return out


def sens():
    out = {}
    for sc in ('S0_walk_icu', 'S1_care_icu'):
        for a in (0.005, 0.01, 0.02, 0.05):
            for l in ('ul', 'dl'):
                runs = A.load(sc, l, 5, a=a)
                runs = {0: runs[0]} if 0 in runs else runs
                if runs:
                    for p in ('KINELINK', 'OLLA', 'KL coverage score', 'Q-learning', 'Oracle'):
                        out[('alpha', sc, a, l, p)] = A.summary(runs, p)
    for e in (0.01, 0.02, 0.05, 0.1, 0.2):
        for l in ('ul', 'dl'):
            runs = A.load('S0_walk_icu', l, 5, a=0.01, e=e)
            runs = {0: runs[0]} if 0 in runs else runs
            if runs:
                for p in ('KINELINK', 'OLLA'):
                    out[('eta', e, l, p)] = A.summary(runs, p)
    for hh in (1, 2, 5):
        for l in ('ul', 'dl'):
            runs = A.load('S0_walk_icu', l, hh)
            if runs:
                for p in ('KINELINK', 'OLLA', 'KL w/o IMU', 'Persist+ACI', 'Oracle', 'Q-learning'):
                    out[('h', hh, l, p)] = A.summary(runs, p)
    return out


def fig_forecast(FT):
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={'wspace': 0.42})
    names = {'persist': 'Persistence', 'ar': 'AR(10)', 'gru': 'GRU', 'lru_imu': 'LRU+IMU', 'lstm': 'LSTM', 'lstm_imu': 'LSTM+IMU (KINELINK)'}
    sty = {'persist': ('-', 'o', '#9a9a9a'), 'ar': ((0, (1, 1)), 's', 'black'), 'gru': ((0, (5, 2)), '^', '#5a5a5a'),
           'lru_imu': ((0, (3, 1, 1, 1)), 'v', 'black'), 'lstm': ('-', 'D', '#5a5a5a'), 'lstm_imu': ('-', 'o', 'black')}
    for ax, l in zip(axs, ('ul', 'dl')):
        for k, (ls, mk, c) in sty.items():
            y = [FT[('S0_walk_icu', l, h, k)][0] for h in (1, 2, 5)]
            ax.plot([10, 20, 50], y, ls=ls, marker=mk, ms=3, color=c, lw=0.9 if k != 'lstm_imu' else 1.3, label=names[k])
        ax.set_xticks([10, 20, 50]); ax.set_xlabel('horizon $D$ (ms)')
        ax.set_title('(a) infrared uplink' if l == 'ul' else '(b) visible-light downlink', fontsize=7.2)
        ax.grid(True, lw=0.3, color='#d0d0d0')
    axs[0].set_ylabel(r'RMSE of $\Delta z_{t+D}$ (dB)')
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='upper center', ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, 1.24), columnspacing=0.8, handlelength=2.2)
    fig.savefig(f'{FIG}/fig4_forecast.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


LAB = {'KINELINK': 'KINELINK', 'GRU+ACI': 'GRU + tracker', 'LRU-IMU+ACI': 'LRU-IMU + tracker', 'AR+ACI': 'AR + tracker',
       'Persist+ACI': 'Persistence + tracker', 'OLLA': 'OLLA', 'LSTM+split': 'LSTM + split margin', 'Q-learning': 'Q-learning',
       'KL coverage score': 'SINR-coverage score', 'KL gain score': 'DC-gain score', 'KL split CP': 'static margin',
       'KL fixed CP': 'fixed prefix'}


def fig_tradeoff(T):
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw={'wspace': 0.45})
    marks = {'KINELINK': ('o', 'black', 6), 'GRU+ACI': ('^', '#555555', 4), 'LRU-IMU+ACI': ('v', '#555555', 4),
             'AR+ACI': ('s', '#555555', 4), 'Persist+ACI': ('D', '#999999', 4), 'OLLA': ('P', '#999999', 5),
             'LSTM+split': ('X', 'black', 5), 'Q-learning': ('*', 'black', 6), 'KL coverage score': ('h', '#777777', 4),
             'KL gain score': ('<', '#777777', 4), 'KL split CP': ('>', '#777777', 4)}
    for ax, l in zip(axs, ('ul', 'dl')):
        for p, (mk, c, s) in marks.items():
            d = T[('S0_walk_icu', l, p)]
            ax.plot(100 * d['viol'][0], d['se'][0], mk, color=c, ms=s, mec='black', mew=0.4, label=LAB[p])
        ax.axvline(1.0, color='black', lw=0.7, ls=(0, (3, 2)))
        orc = T[('S0_walk_icu', l, 'Oracle')]['se'][0]
        ax.axhline(orc, color='#888888', lw=0.6, ls=(0, (1, 1)))
        ax.set_xlabel('violation rate $\\hat{v}$ (%)')
        ax.set_title('(a) uplink' if l == 'ul' else '(b) downlink', fontsize=7.2)
        ax.grid(True, lw=0.3, color='#d0d0d0')
    axs[0].set_ylabel(r'$\overline{\mathrm{SE}}$ (bit/s/Hz)')
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='upper center', ncol=3, fontsize=5.9, bbox_to_anchor=(0.5, 1.3), columnspacing=0.6, handletextpad=0.2)
    fig.savefig(f'{FIG}/fig5_tradeoff.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_dynamics():
    runs = A.load('S1_care_icu', 'ul', 5)
    d = runs[0]['out']
    nd = 'chest'
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.6), sharex=True, gridspec_kw={'hspace': 0.12})
    W = 3000
    sty = {'KINELINK': ('-', 'black', 1.0), 'KL split CP': ((0, (4, 2)), '#555555', 0.9), 'LSTM+split': ((0, (1, 1)), 'black', 0.9),
           'Q-learning': ('-', '#9a9a9a', 0.9)}
    for p, (ls, c, lw) in sty.items():
        x = d[(nd, p)]
        m = ~np.isnan(x['se'])
        v = np.nan_to_num(x['va'], nan=0.0)
        se = np.nan_to_num(x['se'], nan=0.0)
        k = np.ones(W)
        vr = np.convolve(v * m, k, 'valid') / np.maximum(np.convolve(m.astype(float), k, 'valid'), 1)
        sr = np.convolve(se * m, k, 'valid') / np.maximum(np.convolve(m.astype(float), k, 'valid'), 1)
        tt = (np.arange(len(vr)) + W) / 100 / 60
        axs[0].plot(tt, 100 * vr, ls=ls, color=c, lw=lw, label={'KL split CP': 'KINELINK, static margin'}.get(p, LAB.get(p, p)))
        axs[1].plot(tt, sr, ls=ls, color=c, lw=lw)
    axs[0].axhline(1.0, color='black', lw=0.6, ls=(0, (3, 2)))
    axs[0].set_ylabel('violation rate (%)\n30 s window')
    axs[1].set_ylabel('SE (bit/s/Hz)\n30 s window')
    axs[1].set_xlabel('time in the S1 stream (min)')
    for ax in axs:
        ax.grid(True, lw=0.3, color='#d0d0d0')
    axs[0].legend(ncol=2, fontsize=6.2, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    fig.savefig(f'{FIG}/fig6_dynamics.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_bursts():
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.7), gridspec_kw={'wspace': 0.4})
    sty = {'KINELINK': ('-', 'black', 1.1), 'KL fixed CP': ((0, (4, 2)), 'black', 0.9), 'OLLA': ((0, (1, 1)), '#555555', 0.9)}
    for ax, sc in zip(axs, ('S0_walk_icu', 'S1_care_icu')):
        runs = A.load(sc, 'dl', 5)
        for p, (ls, c, lw) in sty.items():
            b = []
            for s, d in runs.items():
                for nd in F.NODES:
                    x = d['out'][(nd, p)]
                    m = ~np.isnan(x['se'])
                    b += A.bursts(x['va'][m])
            b = np.sort(np.array(b))
            k = np.arange(1, 61)
            cc = [(b >= kk).mean() for kk in k]
            ax.step(k, cc, where='post', ls=ls, color=c, lw=lw, label=LAB.get(p, p))
        ax.set_yscale('log'); ax.set_xlabel('burst length $\\ell$ (frames)')
        ax.set_title('(a) S0 walk, downlink' if sc == 'S0_walk_icu' else '(b) S1 care, downlink', fontsize=7.0)
        ax.grid(True, lw=0.3, color='#d0d0d0', which='both')
        ax.set_xlim(1, 60)
    axs[0].set_ylabel(r'$\Pr[\mathrm{burst}\geq\ell]$')
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='upper center', ncol=3, fontsize=6.4, bbox_to_anchor=(0.5, 1.1))
    fig.savefig(f'{FIG}/fig7_bursts.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def fig_sens(S):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw={'wspace': 0.38})
    al = (0.005, 0.01, 0.02, 0.05)
    sty = {'KINELINK': ('-', 'o', 'black'), 'OLLA': ((0, (4, 2)), 's', '#555555'), 'KL coverage score': ((0, (1, 1)), '^', 'black'),
           'Q-learning': ('-', '*', '#9a9a9a')}
    for p, (ls, mk, c) in sty.items():
        y = [S[('alpha', 'S0_walk_icu', a, 'ul', p)]['se'][0] for a in al if ('alpha', 'S0_walk_icu', a, 'ul', p) in S]
        x = [100 * a for a in al if ('alpha', 'S0_walk_icu', a, 'ul', p) in S]
        axs[0].plot(x, y, ls=ls, marker=mk, ms=3.2, color=c, lw=0.9, label=LAB.get(p, p))
        y = [100 * S[('alpha', 'S0_walk_icu', a, 'ul', p)]['viol'][0] for a in al if ('alpha', 'S0_walk_icu', a, 'ul', p) in S]
        axs[1].plot(x, y, ls=ls, marker=mk, ms=3.2, color=c, lw=0.9)
    axs[1].plot([0.5, 5], [0.5, 5], color='#aaaaaa', lw=0.6, ls=(0, (3, 2)))
    axs[0].set_xscale('log'); axs[1].set_xscale('log'); axs[1].set_yscale('log')
    for ax in axs[:2]:
        ax.set_xticks([0.5, 1, 2, 5]); ax.set_xticklabels(['0.5', '1', '2', '5'])
        ax.set_xlabel(r'target $\alpha$ (%)')
    axs[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5]); axs[1].set_yticklabels(['0.1', '0.2', '0.5', '1', '2', '5'])
    from matplotlib.ticker import NullFormatter
    for ax in axs[:2]:
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    axs[0].set_ylabel('uplink SE (bit/s/Hz)'); axs[1].set_ylabel(r'realised $\hat{v}$ (%)')
    axs[0].set_title('(a) efficiency vs target', fontsize=7.0); axs[1].set_title('(b) realised vs target', fontsize=7.0)
    et = (0.01, 0.02, 0.05, 0.1, 0.2)
    for l, ls, mk in (('ul', '-', 'o'), ('dl', (0, (4, 2)), 's')):
        y = [S[('eta', e, l, 'KINELINK')]['se'][0] / S[('eta', 0.05, l, 'KINELINK')]['se'][0] for e in et if ('eta', e, l, 'KINELINK') in S]
        v = [100 * S[('eta', e, l, 'KINELINK')]['viol'][0] for e in et if ('eta', e, l, 'KINELINK') in S]
        x = [e for e in et if ('eta', e, l, 'KINELINK') in S]
        axs[2].plot(x, y, ls=ls, marker=mk, ms=3.2, color='black', lw=0.9, label=f"{'uplink' if l == 'ul' else 'downlink'} SE (rel.)")
        ax2 = axs[2].twinx() if l == 'ul' else ax2
        ax2.plot(x, v, ls=ls, marker=mk, ms=3.2, color='#9a9a9a', lw=0.9, mfc='white')
    ax2.set_ylabel(r'$\hat{v}$ (%) (grey)', fontsize=7)
    axs[2].set_xscale('log'); axs[2].set_xlabel(r'step size $\eta$'); axs[2].set_ylabel('SE relative to $\\eta=0.05$')
    axs[2].set_title('(c) tracker step size (S0)', fontsize=7.0)
    axs[2].legend(fontsize=6.0, loc='lower left')
    for ax in axs:
        ax.grid(True, lw=0.3, color='#d0d0d0')
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='upper center', ncol=4, fontsize=6.4, bbox_to_anchor=(0.36, 1.1))
    fig.savefig(f'{FIG}/fig8_sensitivity.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def main():
    R = {}
    R['corpus'] = json.load(open(f'{RES}/corpus_stats.json'))
    R['val'] = val_losses()
    R['FT'] = forecast_table()
    R['T5'] = la_tables(5)
    R['T2'] = la_tables(2)
    R['ST'] = stats_table(5)
    R['S'] = sens()
    pickle.dump(R, open(f'{RES}/R.pkl', 'wb'))
    fig_forecast(R['FT']); fig_tradeoff(R['T5']); fig_dynamics(); fig_bursts(); fig_sens(R['S'])


def figs_only():
    R = pickle.load(open(f'{RES}/R.pkl', 'rb'))
    fig_forecast(R['FT']); fig_tradeoff(R['T5']); fig_dynamics(); fig_bursts(); fig_sens(R['S'])
    print('ok')


if __name__ == '__main__':
    main()
