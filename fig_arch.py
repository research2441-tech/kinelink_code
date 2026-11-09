"""Fig. 1: KINELINK architecture (grayscale block diagram)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.6, 'mathtext.fontset': 'dejavuserif'})


def box(ax, x, y, w, h, text, fc='white', lw=0.9, ls='-', fs=7.6, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='black', lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal',
            linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, ls='-', lw=0.9, cs='arc3,rad=0', text=None, toff=(0, 0.06), ha='center', fs=6.8):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=8, lw=lw, ls=ls, color='black', connectionstyle=cs,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        ax.text((p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1], text, ha=ha, va='bottom', fontsize=fs,
                style='italic')


def poly(ax, pts, ls='-', lw=0.9):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs + (pts[-1][0],), ys + (pts[-1][1],), color='black', lw=lw, ls=ls) if False else None
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=lw, ls=ls, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw)


fig, ax = plt.subplots(figsize=(7.16, 4.5))
ax.set_xlim(0, 20); ax.set_ylim(0, 12.2); ax.axis('off')
FS = 6.6


def frame(x, y, w, h, title, fc, bottom=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.15', fc=fc, ec='#666666', lw=0.7, ls=(0, (4, 3))))
    ax.text(x + 0.25, (y + 0.42) if bottom else (y + h - 0.32), title, fontsize=7.0, weight='bold', va='center')


frame(0.15, 6.4, 8.15, 5.6, 'Body node (wearable)', '#f3f3f3')
frame(9.3, 6.4, 10.55, 5.6, 'Link controller (access point)', '#f3f3f3')
frame(0.15, 0.2, 8.15, 5.6, 'Offline training', 'white', bottom=True)
frame(9.3, 0.2, 10.55, 5.6, 'Optical channel and online calibration', 'white', bottom=True)

B = lambda *a, **k: box(ax, *a, fs=FS, **k)
B(0.6, 9.2, 3.2, 1.5, 'IMU stream\n' + r'$\boldsymbol{\omega}_t,\ \mathbf{f}_t$')
B(0.6, 6.95, 3.2, 1.6, 'Pilot estimates\n' + r'$z_t,\ \tau_t,\ |H_{g,t}|,\ \varepsilon_t(L)$')
B(4.6, 7.9, 3.4, 2.4, 'Encoder +\nLSTM forecaster\n' + r'$(\mathbf{h}_t,\mathbf{c}_t)$', fc='#d9d9d9')
B(9.8, 9.5, 2.9, 1.7, 'Nominal SINR\n' + r'$\hat{\gamma}_{g}(L)$' + ' at ' + r'$t+D$')
B(9.8, 6.95, 2.9, 1.7, 'ISI extrapolation\nwith ' + r'$\hat{\tau}^{\mathrm{up}}_{t+D}$')
B(13.4, 7.9, 2.9, 2.6, 'Joint CP and\nbit loading,\nmargin ' + r'$e^{-q_t}$', fc='#d9d9d9')
B(17.1, 8.3, 2.4, 1.8, 'DCO-OFDM\nframe ' + r'$t+D$')
B(17.1, 2.9, 2.4, 1.8, 'Optical\nchannel')
B(13.3, 2.9, 3.1, 1.8, 'Violation flag\n' + r'$V_{t+D}$' + ', ' + r'$\tau_{t+D}$')
B(9.6, 2.9, 3.2, 1.8, 'Quantile trackers\n' + r'$q_t,\ q^{\tau}_t$', fc='#d9d9d9')
B(9.8, 1.0, 9.7, 1.3, 'Frame-violation rate ' + r'$\leq\alpha+(\bar{q}-q_1)/(\eta T)+(2D+1)/T$' + '\nfor any channel sequence')
B(0.5, 3.2, 2.2, 1.8, 'CMU motion\ncapture')
B(3.15, 3.2, 2.2, 1.8, 'Ray-traced\nICU / FTPR')
B(5.75, 3.2, 2.4, 1.8, 'Multi-horizon\nloss ' + r'$\mathcal{L}(\Theta)$')
B(3.15, 0.95, 5.0, 1.2, 'AdamW update of ' + r'$\Theta$')

lab = lambda x, y, t, ha='center', va='center': ax.text(x, y, t, fontsize=6.3, style='italic', ha=ha, va=va)
# body node
arrow(ax, (3.8, 9.95), (4.6, 9.45))
arrow(ax, (3.8, 7.75), (4.6, 8.45))
arrow(ax, (8.0, 9.8), (9.8, 10.35))
ax.text(8.75, 10.5, r'$\Delta\hat{z},\ \hat{\tau}$', fontsize=6.3, style='italic', ha='center', va='bottom',
        bbox=dict(fc='white', ec='none', pad=0.6))
poly(ax, [(2.2, 6.95), (2.2, 6.65), (9.05, 6.65), (9.05, 7.8), (9.8, 7.8)])
lab(6.4, 6.72, r'$\varepsilon_t(L),\ \tau_t$', va='bottom')
# controller
arrow(ax, (12.7, 10.35), (13.4, 9.75))
arrow(ax, (12.7, 7.8), (13.4, 8.55))
arrow(ax, (16.3, 9.2), (17.1, 9.2))
arrow(ax, (18.3, 8.3), (18.3, 4.7))
arrow(ax, (17.1, 3.8), (16.3, 3.8))
arrow(ax, (13.4, 3.8), (12.7, 3.8))
poly(ax, [(11.9, 4.7), (11.9, 6.1), (14.2, 6.1), (14.2, 7.9)])
lab(14.3, 7.05, r'$q_t$', ha='left')
poly(ax, [(10.4, 4.7), (10.4, 6.95)])
lab(10.5, 5.2, r'$q^{\tau}_t$', ha='left')
poly(ax, [(15.6, 7.9), (15.6, 4.7)], ls=(0, (3, 2)))
lab(15.7, 5.15, 'stored\nloading', ha='left')
arrow(ax, (11.2, 2.9), (11.2, 2.3))
# offline
arrow(ax, (2.7, 4.1), (3.15, 4.1))
arrow(ax, (5.35, 4.1), (5.75, 4.1))
poly(ax, [(6.95, 3.2), (6.95, 2.15)])
poly(ax, [(7.6, 5.0), (7.6, 7.9)], ls=(0, (4, 2)))
lab(7.45, 7.3, r'$\nabla_{\Theta}\mathcal{L}$ (BPTT)', ha='right')
poly(ax, [(3.15, 1.55), (0.32, 1.55), (0.32, 6.1), (5.2, 6.1), (5.2, 7.9)])
lab(2.6, 6.15, r'$\Theta^{\star}$', va='bottom')

fig.savefig('./figs/fig1_architecture.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
print('saved')
