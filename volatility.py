"""Kinematics-conditioned volatility model: predicts the scale of the forecast error from inertial
intensity and recent channel/forecast behaviour (causal features only)."""
import numpy as np

EPS = 0.05


def _rms_past(x, w):
    """Causal rolling RMS over the previous w samples (excluding the current one)."""
    c = np.concatenate([[0.0], np.cumsum(x ** 2)])
    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        a = max(0, t - w)
        n = t - a
        out[t] = np.sqrt((c[t] - c[a]) / n) if n > 0 else np.nan
    return out


def features(U, dz_hat, z_meas, h, imu=True, w=20):
    """U: model inputs (scaled IMU in columns 2..7); dz_hat: forecast of z(t+h)-z_meas(t)."""
    T = len(z_meas)
    gyro = np.linalg.norm(U[:, 2:5] * 5, axis=1)
    acc = np.abs(np.linalg.norm(U[:, 5:8] * 10, axis=1) - 9.81)
    dzm = np.concatenate([[0.0], np.diff(z_meas)])
    vol = _rms_past(dzm, w)
    # realised forecast errors become known h frames later
    err = np.full(T, np.nan)
    err[h:] = (z_meas[h:] - z_meas[:-h]) - dz_hat[:-h]
    e_known = np.concatenate([np.full(h, np.nan), err[:-h]]) if h > 0 else err
    e_known = np.nan_to_num(e_known, nan=0.0)
    evol = _rms_past(e_known, w)
    vol = np.nan_to_num(vol, nan=0.3); evol = np.nan_to_num(evol, nan=0.3)
    cols = [np.ones(T), np.log(vol + EPS), np.log(evol + EPS)]
    if imu:
        cols += [np.log1p(gyro), np.log1p(acc)]
    return np.stack(cols, 1), err


class VolModel:
    def __init__(self, imu=True, lam=1e-2):
        self.imu, self.lam = imu, lam

    def fit(self, Xs, es):
        X = np.concatenate(Xs); e = np.concatenate(es)
        m = ~np.isnan(e)
        X, y = X[m], np.log(np.abs(e[m]) + EPS)
        self.W = np.linalg.solve(X.T @ X + self.lam * np.eye(X.shape[1]), X.T @ y)
        self.c = np.median(np.abs(e[m]) / np.exp(X @ self.W))
        return self

    def predict(self, X):
        return self.c * np.exp(X @ self.W)
