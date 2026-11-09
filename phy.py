"""DCO-OFDM physical layer: clipping, receiver noise, group SINR, M-QAM SER and bit loading."""
import numpy as np
from scipy.special import erfc
from scipy.stats import norm
import raytrace as rt

Q_E, K_B, TEMP = 1.602e-19, 1.381e-23, 295.0


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


BITS = np.array([0, 1, 2, 4, 6, 8])


def ser(bits, g):
    """Symbol error rate of BPSK / square M-QAM at SNR g (linear)."""
    g = np.maximum(g, 0)
    out = np.ones(np.broadcast(bits, g).shape)
    b = np.broadcast_to(bits, out.shape); gg = np.broadcast_to(g, out.shape)
    m1 = b == 1
    out[m1] = qfunc(np.sqrt(2 * gg[m1]))
    for k in (2, 4, 6, 8):
        mk = b == k
        M = 2 ** k
        p = 2 * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * gg[mk] / (M - 1)))
        out[mk] = 1 - (1 - p) ** 2
    out[b == 0] = 0.0
    return out


def snr_threshold(ser_tar):
    """Minimum linear SNR at which each constellation meets the target SER (bisection)."""
    th = np.zeros(len(BITS))
    for i, b in enumerate(BITS):
        if b == 0:
            continue
        lo, hi = 1e-3, 1e7
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if ser(np.array([b]), np.array([mid]))[0] > ser_tar:
                lo = mid
            else:
                hi = mid
        th[i] = hi
    return th


def clipping(kappa):
    """Bussgang gain and clipping-noise variance (normalised to unit signal variance) for double-sided
    clipping at +/- kappa standard deviations (LED linear range around the illumination bias)."""
    x = np.linspace(-12, 12, 200001)
    pdf = norm.pdf(x)
    y = np.clip(x, -kappa, kappa)
    K = np.trapezoid(x * y * pdf, x)
    var_y = np.trapezoid(y ** 2 * pdf, x)
    return K, var_y - K ** 2


class Link:
    def __init__(self, link):
        cfg = rt.LINK[link]
        self.link = link
        self.n_fft, self.bw, self.groups, self.cp_opts = cfg['n_fft'], cfg['bw'], cfg['groups'], np.array(cfg['cp_opts'])
        k = np.arange(1, self.n_fft // 2)
        self.gsize = np.array([len(c) for c in np.array_split(k, self.groups)])
        self.f = rt.group_freqs(self.n_fft, self.bw, self.groups)
        if link == 'dl':
            self.P_avg, self.R, self.A, self.I_bg, self.f_led = 2.0, 0.50, 1e-4, 50e-6, 60e6
            self.n_tx = None
        else:
            self.P_avg, self.R, self.A, self.I_bg, self.f_led = 0.030, 0.60, 2e-4, 100e-6, 50e6
        self.kappa = 3.2
        self.sig = self.P_avg / self.kappa                      # optical AC standard deviation
        self.K, self.clip_var = clipping(self.kappa)
        self.led2 = 1.0 / (1.0 + (self.f / self.f_led) ** 2)  # first-order LED response |H_LED|^2

    def noise_var(self, H0):
        """Shot (signal DC + background) plus thermal noise current variance (Komine-Nakagawa model)."""
        B = self.bw
        eta, I2, I3, G, Gam, gm = 112e-12 / 1e-4, 0.562, 0.0868, 10.0, 1.5, 30e-3
        ca = eta * self.A
        P_dc = self.P_avg * H0          # H0 aggregates all luminaires for the downlink
        shot = 2 * Q_E * (self.R * P_dc + self.I_bg * I2) * B
        th = 8 * np.pi * K_B * TEMP * ca * I2 * B ** 2 / G + 16 * np.pi ** 2 * K_B * TEMP * Gam * ca ** 2 * I3 * B ** 3 / gm
        return shot + th

    def sinr(self, Hg, H0, isi_frac):
        """Group SINR. Hg: (..., groups) |H| at group centres; isi_frac: tap-energy fraction beyond the CP."""
        s = (self.R * self.K * self.sig) ** 2 * Hg ** 2 * self.led2
        clip = (self.R * self.sig) ** 2 * self.clip_var * Hg ** 2 * self.led2
        isi = 2 * isi_frac[..., None] * (self.R * self.K * self.sig) ** 2 * (Hg ** 2 * self.led2).mean(-1, keepdims=True)
        return s / (self.noise_var(H0)[..., None] + clip + isi)

    def load(self, g, th):
        """Largest constellation per group meeting the SER target at SNR g; returns bits per group."""
        idx = (g[..., None] >= th[None, 1:]).sum(-1)
        return BITS[idx]

    def se(self, bits, L):
        return (bits * self.gsize).sum(-1) / (self.n_fft + L)
