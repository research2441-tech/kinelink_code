"""Fig. 2 (room layouts with trajectories) and Fig. 3 (example channel traces)."""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rooms

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'legend.frameon': False, 'mathtext.fontset': 'dejavuserif'})
DATA = './data'


def layouts():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={'width_ratios': [11.5, 7.0], 'wspace': 0.18})
    for ax, room, pref in ((axs[0], rooms.icu(), ('S0_walk_icu', 'S4_walk_icu_crowded')), (axs[1], rooms.ftpr(), ('S3_walk_ftpr',))):
        ax.add_patch(Rectangle((0, 0), room['L'], room['W'], fill=False, lw=1.0))
        for b in room['boxes']:
            hatch = '////' if b in room['tall'] else ('....' if b[5] < 1.0 else 'xxxx')
            ax.add_patch(Rectangle((b[0], b[1]), b[3] - b[0], b[4] - b[1], fc='#e6e6e6', ec='black', lw=0.5, hatch=hatch))
        files = sorted(glob.glob(f'{DATA}/{pref[0]}__*.npz'))[:8]
        for i, f in enumerate(files):
            d = np.load(f)
            ax.plot(d['hips'][:, 0], d['hips'][:, 1], color='black', lw=0.8, alpha=0.85)
            ax.plot(d['hips'][0, 0], d['hips'][0, 1], 'o', ms=2.2, color='black')
            if i < 2:
                for s in range(d['staff'].shape[0]):
                    ax.plot(d['staff'][s, :, 0], d['staff'][s, :, 1], color='#7f7f7f', lw=0.7, ls=(0, (3, 2)))
        ax.plot(room['lum'][:, 0], room['lum'][:, 1], 's', ms=5.5, mfc='white', mec='black', mew=0.8)
        ax.plot(room['lum'][:, 0], room['lum'][:, 1], '+', ms=4.5, color='black', mew=0.8)
        ax.set_xlim(-0.2, room['L'] + 0.2); ax.set_ylim(-0.2, room['W'] + 0.2); ax.set_aspect('equal')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f"({'a' if room['name'] == 'ICU' else 'b'}) {room['name']}: {room['L']} m × {room['W']} m × {room['H']} m, "
                     f"{len(room['lum'])} luminaires", fontsize=7.5)
    h = [plt.Line2D([], [], color='black', lw=0.8, label='wearer (CMU clip, hips)'),
         plt.Line2D([], [], color='#7f7f7f', lw=0.7, ls=(0, (3, 2)), label='moving staff'),
         plt.Line2D([], [], marker='s', mfc='white', mec='black', ls='none', ms=5, label='luminaire + IR receiver'),
         Rectangle((0, 0), 1, 1, fc='#e6e6e6', ec='black', hatch='....', lw=0.5, label='bed / sofa'),
         Rectangle((0, 0), 1, 1, fc='#e6e6e6', ec='black', hatch='xxxx', lw=0.5, label='nurse desk'),
         Rectangle((0, 0), 1, 1, fc='#e6e6e6', ec='black', hatch='////', lw=0.5, label='bathroom (blocks rays)')]
    fig.legend(handles=h, loc='lower center', ncol=6, fontsize=6.6, bbox_to_anchor=(0.5, -0.13), handlelength=1.8, columnspacing=1.2)
    fig.savefig('./figs/fig2_rooms.png', dpi=400, bbox_inches='tight', pad_inches=0.03)


def traces():
    # longest S0 clip
    files = sorted(glob.glob(f'{DATA}/S0_walk_icu__*.npz'), key=lambda f: -len(np.load(f)['t']))
    d = np.load(files[0])
    t = d['t']
    w = (t >= 20) & (t <= 40)
    fig, axs = plt.subplots(3, 1, figsize=(3.45, 3.9), sharex=True, gridspec_kw={'hspace': 0.12})
    styles = {'shoulder': ('-', 'black'), 'chest': ((0, (4, 1.5)), 'black'), 'wrist': ('-', '#8c8c8c'), 'ankle': ((0, (1, 1)), 'black')}
    for nd, (ls, c) in styles.items():
        axs[0].plot(t[w], (10 * np.log10(d[f'{nd}_ul_H0']))[w], ls=ls, color=c, lw=0.8, label=nd)
        axs[1].plot(t[w], (10 * np.log10(d[f'{nd}_dl_H0']))[w], ls=ls, color=c, lw=0.8)
        axs[2].plot(t[w], (d[f'{nd}_dl_tau'] * 1e9)[w], ls=ls, color=c, lw=0.8)
    axs[0].set_ylabel('UL gain $H_0$ (dB)')
    axs[1].set_ylabel('DL gain $H_0$ (dB)')
    axs[2].set_ylabel(r'DL $\tau_{\mathrm{RMS}}$ (ns)')
    axs[2].set_xlabel('time (s)')
    for ax in axs:
        ax.grid(True, lw=0.3, color='#cccccc')
    axs[0].legend(ncol=4, fontsize=6.4, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=0.9)
    clip = files[0].split('__')[1]
    fig.savefig('./figs/fig3_traces.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
    return clip, float(t[-1])


if __name__ == '__main__':
    layouts()
    print(traces())
