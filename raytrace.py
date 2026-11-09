"""Site-specific optical channel computation for body-worn nodes.

Downlink: ceiling luminaires (VLC) -> photodiode on a body segment.
Uplink:   infrared emitter on a body segment -> ceiling photodiodes (selection combining).
Paths: line of sight (with body/staff shadowing), first-order diffuse reflections from
discretised room surfaces, and a higher-order diffuse tail from the integrating-sphere model.
All gains are optical power ratios for unit transmitted optical power.
"""
import numpy as np

C0 = 299792458.0
DT = 0.25e-9                  # histogram bin (s)
NB = 800                      # 200 ns window


class Optics:
    m_lum = 1.0                         # luminaire Lambertian order (60 deg semi-angle)
    A_rx, fov_rx, n_rx = 1e-4, np.deg2rad(70.0), 1.5     # body photodiode (downlink)
    m_tx = 2.0                          # body IR emitter (45 deg semi-angle)
    A_ap, fov_ap, n_ap = 2e-4, np.deg2rad(60.0), 1.5     # ceiling IR photodiodes (uplink)


def conc_gain(n, fov):
    return n ** 2 / np.sin(fov) ** 2


def precompute(room, pc):
    c, nrm, area, rho = pc
    lum = room['lum']
    m = Optics.m_lum
    d = c[:, None, :] - lum[None, :, :]                       # P x S x 3
    dist = np.linalg.norm(d, axis=2)
    cphi = np.clip(-d[..., 2] / dist, 0, None)                # luminaire faces -z
    cth = np.clip(-(d * nrm[:, None, :]).sum(2) / dist, 0, None)
    w = (m + 1) / (2 * np.pi * dist ** 2) * cphi ** m * cth * area[:, None]
    P = w.sum(1)
    mu = (w * dist).sum(1) / np.maximum(P, 1e-30) / C0
    sd = np.sqrt(np.maximum((w * (dist / C0 - mu[:, None]) ** 2).sum(1) / np.maximum(P, 1e-30), 0))
    # uplink: patch -> ceiling receiver static gains
    ap = lum.copy()
    g_ap = conc_gain(Optics.n_ap, Optics.fov_ap)
    dk = ap[None, :, :] - c[:, None, :]
    dkn = np.linalg.norm(dk, axis=2)
    cth_k = np.clip((dk * nrm[:, None, :]).sum(2) / dkn, 0, None)
    cpsi_k = np.clip(dk[..., 2] / dkn, 0, None)          # receiver faces -z; incidence from the patch
    vis = cpsi_k >= np.cos(Optics.fov_ap)
    S = rho[:, None] * Optics.A_ap * cth_k * cpsi_k * g_ap / (np.pi * dkn ** 2) * vis
    A_room = area.sum()
    V = room['L'] * room['W'] * room['H']
    rbar = (rho * area).sum() / A_room
    mfp = 4 * V / A_room
    tau_d = -mfp / (C0 * np.log(rbar))
    return dict(c=c, n=nrm, area=area, rho=rho, Pinc=P, mu=mu, sd=sd, S=S, dk=dkn / C0,
                A_room=A_room, rbar=rbar, mfp=mfp, tau_d=tau_d, ap=ap)


def seg_seg_dist(p1, q1, p2, q2):
    """Minimum distance between segments [p1,q1] and [p2,q2] (broadcast over leading dims)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    b = (d1 * d2).sum(-1); cc = (d1 * r).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - cc * e) / np.where(den > 1e-12, den, 1), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    s = np.where(t < 0, np.clip(-cc / np.maximum(a, 1e-12), 0, 1), np.where(t > 1, np.clip((b - cc) / np.maximum(a, 1e-12), 0, 1), s))
    t = np.clip(t, 0, 1)
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def blocked(r, s, cyl_a, cyl_b, cyl_r, boxes):
    """r: R x 3 node points; s: S x 3 or R x S x 3 far ends; cylinders R x C x 3 endpoints."""
    if s.ndim == 2:
        s = np.broadcast_to(s[None], (r.shape[0],) + s.shape)
    p1 = r[:, None, None, :]
    q1 = s[:, :, None, :]
    dist = seg_seg_dist(p1, q1, cyl_a[:, None, :, :], cyl_b[:, None, :, :])
    blk = (dist < cyl_r[None, None, :]).any(-1)
    for b in boxes:                     # tall partitions: slab test
        lo, hi = np.array(b[:3]), np.array(b[3:])
        dvec = s - r[:, None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo - r[:, None, :]) / dvec
            t2 = (hi - r[:, None, :]) / dvec
        tmin = np.nanmax(np.minimum(t1, t2), -1)
        tmax = np.nanmin(np.maximum(t1, t2), -1)
        blk |= (tmax >= np.maximum(tmin, 0)) & (tmin <= 1) & (tmax > 1e-6) & (tmin < 1 - 1e-6)
    return blk


def _hist(R, tt, ww):
    idx = np.clip((tt / DT).astype(np.int64), 0, NB - 1)
    rows = np.repeat(np.arange(R), idx.shape[1])
    h = np.bincount(rows * NB + idx.ravel(), weights=ww.ravel(), minlength=R * NB)
    return h.reshape(R, NB)


def _tail(eta, t_s, tau_d):
    t = (np.arange(NB) + 0.5) * DT
    prof = np.where(t > t_s, np.exp(-(t - t_s) / tau_d), 0.0)
    prof = prof / max(prof.sum(), 1e-30)
    return eta * prof


def downlink(room, pre, r, nr, cyl_a, cyl_b, cyl_r):
    R = r.shape[0]
    lum = room['lum']
    g = conc_gain(Optics.n_rx, Optics.fov_rx)
    # LOS
    d = lum[None, :, :] - r[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    cphi = np.clip(d[..., 2] / dist, 0, None)
    cpsi = (d * nr[:, None, :]).sum(2) / dist
    vis = cpsi >= np.cos(Optics.fov_rx)
    blk = blocked(r, lum, cyl_a, cyl_b, cyl_r, room['tall'])
    m = Optics.m_lum
    wl = (m + 1) * Optics.A_rx / (2 * np.pi * dist ** 2) * cphi ** m * np.clip(cpsi, 0, None) * g * vis * (~blk)
    tl = dist / C0
    # first-order diffuse
    dp = pre['c'][None, :, :] - r[:, None, :]
    dpn = np.linalg.norm(dp, axis=2)
    cth = np.clip(-(dp * pre['n'][None]).sum(2) / dpn, 0, None)
    cps = (dp * nr[:, None, :]).sum(2) / dpn
    v2 = cps >= np.cos(Optics.fov_rx)
    wd = pre['rho'][None] * pre['Pinc'][None] * Optics.A_rx * cth * np.clip(cps, 0, None) * g / (np.pi * dpn ** 2) * v2
    base = dpn / C0
    td1 = base + (pre['mu'] - pre['sd'])[None]
    td2 = base + (pre['mu'] + pre['sd'])[None]
    tt = np.concatenate([tl, td1, td2], 1)
    ww = np.concatenate([wl, wd / 2, wd / 2], 1)
    h = _hist(R, tt, ww)
    eta = len(lum) * pre['rbar'] ** 2 / (1 - pre['rbar']) / pre['A_room'] * Optics.A_rx * Optics.n_rx ** 2
    t_s = (room['H'] + 2 * pre['mfp']) / C0
    h += _tail(eta, t_s, pre['tau_d'])[None]
    return h, (wl > 0).sum(1)


def uplink(room, pre, r, nr, cyl_a, cyl_b, cyl_r):
    R = r.shape[0]
    ap = pre['ap']
    g = conc_gain(Optics.n_ap, Optics.fov_ap)
    m = Optics.m_tx
    d = ap[None, :, :] - r[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    cphi = np.clip((d * nr[:, None, :]).sum(2) / dist, 0, None)
    cpsi = np.clip(d[..., 2] / dist, 0, None)
    vis = cpsi >= np.cos(Optics.fov_ap)
    blk = blocked(r, ap, cyl_a, cyl_b, cyl_r, room['tall'])
    wl = (m + 1) * Optics.A_ap / (2 * np.pi * dist ** 2) * cphi ** m * cpsi * g * vis * (~blk)
    dp = pre['c'][None, :, :] - r[:, None, :]
    dpn = np.linalg.norm(dp, axis=2)
    cph = np.clip((dp * nr[:, None, :]).sum(2) / dpn, 0, None)
    cth = np.clip(-(dp * pre['n'][None]).sum(2) / dpn, 0, None)
    Pj = (m + 1) / (2 * np.pi * dpn ** 2) * cph ** m * cth * pre['area'][None]
    tot = wl + Pj @ pre['S']
    k = tot.argmax(1)
    wd = Pj * pre['S'][:, k].T
    td = dpn / C0 + pre['dk'][:, k].T
    tl = dist[np.arange(R), k][:, None] / C0
    tt = np.concatenate([tl, td], 1)
    ww = np.concatenate([wl[np.arange(R), k][:, None], wd], 1)
    h = _hist(R, tt, ww)
    eta = pre['rbar'] ** 2 / (1 - pre['rbar']) / pre['A_room'] * Optics.A_ap * Optics.n_ap ** 2
    t_s = (room['H'] + 2 * pre['mfp']) / C0
    h += _tail(eta, t_s, pre['tau_d'])[None]
    return h, k


LINK = {'dl': dict(n_fft=128, bw=50e6, groups=32, cp_opts=(0, 1, 2, 3, 4, 6, 8)),
        'ul': dict(n_fft=64, bw=20e6, groups=31, cp_opts=(0, 1, 2, 3, 4))}


def group_freqs(n_fft, bw, groups):
    k = np.arange(1, n_fft // 2)
    centres = np.array_split(k, groups)
    return np.array([c.mean() for c in centres]) * bw / n_fft


def summarise(h, n_fft=64, bw=200e6, groups=31, cp_opts=(0, 1, 2, 3, 4)):
    """DC gain, RMS delay spread, group-centre subcarrier responses and ISI leakage per CP length."""
    t = (np.arange(NB) + 0.5) * DT
    H0 = h.sum(1)
    mu = (h * t).sum(1) / H0
    tau = np.sqrt(np.maximum((h * t ** 2).sum(1) / H0 - mu ** 2, 0))
    f = group_freqs(n_fft, bw, groups)
    E = np.exp(-2j * np.pi * t[:, None] * f[None, :])
    Hf = h @ E
    # tap-domain response at Ts = 1/bw relative to first significant arrival
    Ts = 1.0 / bw
    first = np.argmax(h > 0.02 * h.max(1, keepdims=True), axis=1)
    ntap = int(np.ceil(NB * DT / Ts)) + 1
    taps = np.zeros((h.shape[0], ntap))
    rel = (np.arange(NB)[None, :] - first[:, None]) * DT
    tap_idx = np.floor(np.maximum(rel, -1e-15) / Ts).astype(int)
    tap_idx = np.where(rel < 0, 0, tap_idx)
    rows = np.repeat(np.arange(h.shape[0]), NB)
    np.add.at(taps, (rows, tap_idx.ravel()), h.ravel())
    e = taps ** 2
    tot = e.sum(1)
    isi = np.stack([e[:, L + 1:].sum(1) / tot for L in cp_opts], 1)
    return dict(H0=H0, tau=tau, Hf=Hf, isi=isi, first=first)
