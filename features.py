"""Turn simulated clips into per-node, per-link sequences of model inputs and multi-horizon targets."""
import glob, os
import numpy as np

NODES = ('shoulder', 'chest', 'wrist', 'ankle')
LINKS = ('dl', 'ul')
HORIZONS = (1, 2, 5)              # frames of 10 ms
DATA = os.environ.get('DATA_DIR', './data')
SIG_Z, SIG_TAU, SIG_GYRO, SIG_ACC = 0.2, 0.03, 0.02, 0.05   # measurement / IMU noise
N_IN = 2 + 6 + len(NODES) + len(LINKS)


def files(split):
    return sorted(glob.glob(f'{DATA}/{split}__*.npz'))


def streams(split, seed=0, imu=True):
    """List of dicts, one per (clip, node, link): inputs U (T x N_IN), targets, raw channel arrays."""
    rng = np.random.default_rng(seed)
    out = []
    for f in files(split):
        d = np.load(f)
        clip = os.path.basename(f).split('__')[1]
        for i, nd in enumerate(NODES):
            gyro = d[f'{nd}_gyro'] + SIG_GYRO * rng.standard_normal(d[f'{nd}_gyro'].shape)
            acc = d[f'{nd}_acc'] + SIG_ACC * rng.standard_normal(d[f'{nd}_acc'].shape)
            for j, l in enumerate(LINKS):
                zt = 10 * np.log10(d[f'{nd}_{l}_H0'])
                tt = d[f'{nd}_{l}_tau'] * 1e9
                zm = zt + SIG_Z * rng.standard_normal(zt.shape)
                tm = tt * (1 + SIG_TAU * rng.standard_normal(tt.shape))
                T = len(zt)
                oh = np.zeros((T, len(NODES) + len(LINKS)))
                oh[:, i] = 1; oh[:, len(NODES) + j] = 1
                U = np.concatenate([((zm + 40) / 10)[:, None], np.log(tm / 10)[:, None],
                                    (gyro / 5) * imu, (acc / 10) * imu, oh], 1).astype(np.float32)
                Y = np.full((T, 2 * len(HORIZONS)), np.nan, np.float32)
                for k, h in enumerate(HORIZONS):
                    Y[:T - h, k] = zt[h:] - zm[:T - h]
                    Y[:T - h, len(HORIZONS) + k] = np.log(tt[h:] / 10)
                out.append(dict(U=U, Y=Y, z_meas=zm, tau_meas=tm, node=nd, link=l, clip=clip, file=f,
                                Hf=d[f'{nd}_{l}_Hf'], H0=d[f'{nd}_{l}_H0'], isi=d[f'{nd}_{l}_isi'], tau=tt))
    return out


def windows(st, L=128, hop=64):
    """Fixed-length training windows with a validity mask (padded at the end of short clips)."""
    Us, Ys, Ms = [], [], []
    for s in st:
        T = len(s['U'])
        starts = list(range(0, max(T - L, 0) + 1, hop)) or [0]
        for a in starts:
            u = np.zeros((L, N_IN), np.float32); y = np.zeros((L, 2 * len(HORIZONS)), np.float32)
            m = np.zeros((L, 2 * len(HORIZONS)), np.float32)
            seg_u, seg_y = s['U'][a:a + L], s['Y'][a:a + L]
            n = len(seg_u)
            u[:n] = seg_u
            ok = ~np.isnan(seg_y)
            y[:n][ok] = seg_y[ok]
            m[:n] = ok
            m[:min(n, 10)] = 0            # short warm-up excluded from the loss
            Us.append(u); Ys.append(y); Ms.append(m)
    return np.stack(Us), np.stack(Ys), np.stack(Ms)
