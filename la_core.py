"""Online link-adaptation engines (numba). One call runs every frame of a scenario stream for a set of
policy rows that share the same link configuration.

Decision at frame t targets frame t+h. Calibration feedback for frame t+h becomes available at t+h,
so the quantile tracker uses outcomes that are h frames old (delayed-feedback form).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _bits_for(g, th, bits_tab):
    b = 0
    for i in range(1, bits_tab.shape[0]):
        if g >= th[i]:
            b = bits_tab[i]
    return b


@njit(cache=True)
def run_calibrated(
        # nominal ingredients at decision time t (per row r): signal/clip/noise terms from the forecast
        S_hat, C_hat, N_hat, Smean_hat, isi_meas, tau_meas, tau_hat,
        # realised ingredients at frame t (true channel)
        S_true, C_true, N_true, Smean_true, isi_true,
        valid, h, alpha, eta, q0, alpha_tau, eta_tau, qtau0, adapt_q, adapt_tau, cp_mode, cp_fix,
        score_mode, th, bits_tab, gsize, n_fft, cp_opts, ser_tar, B, sig):
    """
    S_hat etc: (R, T, G) nominal signal/clip at the decision time for target t+h; N_hat, Smean_hat: (R, T)
    isi_meas: (R, T, nL) measured ISI leakage at decision time; tau_meas, tau_hat: (R, T) seconds
    S_true etc: (R, T, G) realised at frame index t; isi_true: (R, T, nL)
    valid: (R, T) bool, decision at t for t+h allowed (same clip)
    sig: (R, T) additive margin reallocation kappa*(sigma_hat-1) in nats (zeros for a uniform margin)
    score_mode 0: decision-focused group score; 1: scalar gain score (parameter-space ablation);
               2: violation-driven offset (outer-loop link adaptation)
    cp_mode 0: delay-spread-aware CP selection; 1: fixed CP index cp_fix
    Returns per (R, T): se, viol_any, viol_mean, q used, chosen CP index.
    """
    R, T, G = S_hat.shape
    nL = cp_opts.shape[0]
    se = np.full((R, T), np.nan)
    va = np.full((R, T), np.nan)
    vm = np.full((R, T), np.nan)
    qs = np.full((R, T), np.nan)
    Ls = np.full((R, T), -1)
    sc_out = np.full((R, T), np.nan)
    for r in range(R):
        q = q0[r]
        qt = qtau0[r]
        # store decisions to evaluate at t+h
        dec_L = np.zeros(T, np.int64)
        dec_q = np.zeros(T)
        dec_bits = np.zeros((T, G), np.int64)
        dec_ghat = np.zeros((T, G))
        for t in range(T):
            # ---- 1. outcome of decision made at t-h (realised now) ----
            td = t - h
            if td >= 0 and valid[r, td]:
                Li = dec_L[td]
                worst = -1e9
                ok_any = 1
                ser_sum = 0.0
                n_sym = 0.0
                g_true_min = 1e30
                for g in range(G):
                    gt = S_true[r, t, g] / (N_true[r, t] + C_true[r, t, g] + 2.0 * isi_true[r, t, Li] * Smean_true[r, t])
                    b = dec_bits[td, g]
                    if b > 0:
                        # symbol error rate of the loaded constellation
                        if b == 1:
                            p = 0.5 * _erfc(np.sqrt(gt))
                            sr = p
                        else:
                            M = 2.0 ** b
                            p = 2.0 * (1.0 - 1.0 / np.sqrt(M)) * 0.5 * _erfc(np.sqrt(1.5 * gt / (M - 1.0)))
                            sr = 1.0 - (1.0 - p) ** 2
                        if sr > ser_tar:
                            ok_any = 0
                        ser_sum += sr * gsize[g]
                        n_sym += gsize[g]
                    if dec_ghat[td, g] >= th[1]:
                        s = np.log(max(dec_ghat[td, g], 1e-30)) - np.log(max(gt, 1e-30))
                        if s > worst:
                            worst = s
                informative = worst > -1e8          # at least one group was loadable at zero margin
                if score_mode == 1:
                    worst = 0.0
                if worst < -B:
                    worst = -B
                if worst > B:
                    worst = B
                bits_rate = 0.0
                for g in range(G):
                    bits_rate += dec_bits[td, g] * gsize[g]
                se[r, td] = bits_rate / (n_fft + cp_opts[Li])
                va[r, td] = 1.0 - ok_any
                vm[r, td] = 1.0 if (n_sym > 0 and ser_sum / n_sym > ser_tar) else 0.0
                qs[r, td] = dec_q[td]
                Ls[r, td] = Li
                sc_out[r, td] = worst
                if adapt_q[r] == 1:
                    if score_mode == 0:
                        err = 1.0 if worst > dec_q[td] else alpha * (0.0 if informative else 1.0)
                    elif score_mode == 2:          # violation-driven offset (outer-loop link adaptation)
                        err = 1.0 - ok_any
                    else:
                        # scalar score on the DC gain only (computed in caller-provided units: S_hat ratio)
                        sc = np.log(max(Smean_hat[r, td], 1e-300)) - np.log(max(Smean_true[r, t], 1e-300))
                        err = 1.0 if sc > dec_q[td] else 0.0
                    q = q + eta * (err - alpha)
                if adapt_tau[r] == 1:
                    errt = 1.0 if (np.log(tau_meas[r, t]) - np.log(tau_hat[r, td])) > qt else 0.0
                    qt = qt + eta_tau * (errt - alpha_tau)
            # ---- 2. decision at t for frame t+h ----
            if not valid[r, t]:
                continue
            tau_up = tau_hat[r, t] * np.exp(qt)
            best_se = -1.0
            best_L = 0
            for Li in range(nL):
                if cp_mode == 1 and Li != cp_fix:
                    continue
                if cp_mode == 0:
                    lm = isi_meas[r, t, Li]
                    eps = np.exp((tau_meas[r, t] / tau_up) * np.log(max(lm, 1e-30))) if lm > 0 else 0.0
                else:
                    eps = isi_meas[r, t, Li]
                rate = 0.0
                for g in range(G):
                    gh = S_hat[r, t, g] / (N_hat[r, t] + C_hat[r, t, g] + 2.0 * eps * Smean_hat[r, t])
                    rate += _bits_for(gh * np.exp(-(q + sig[r, t])), th, bits_tab) * gsize[g]
                val = rate / (n_fft + cp_opts[Li])
                if val > best_se + 1e-12:
                    best_se = val
                    best_L = Li
            # commit
            if cp_mode == 0:
                lm = isi_meas[r, t, best_L]
                eps = np.exp((tau_meas[r, t] / tau_up) * np.log(max(lm, 1e-30))) if lm > 0 else 0.0
            else:
                eps = isi_meas[r, t, best_L]
            for g in range(G):
                gh = S_hat[r, t, g] / (N_hat[r, t] + C_hat[r, t, g] + 2.0 * eps * Smean_hat[r, t])
                dec_ghat[t, g] = gh
                dec_bits[t, g] = _bits_for(gh * np.exp(-(q + sig[r, t])), th, bits_tab)
            dec_L[t] = best_L
            dec_q[t] = q
    return se, va, vm, qs, Ls, sc_out


@njit(cache=True)
def _erfc(x):
    # Abramowitz-Stegun 7.1.26 rational approximation is too coarse at small error rates;
    # use the complementary error function via continued exp form (W. J. Cody style, double precision)
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
          t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))))
    return ans if x >= 0 else 2.0 - ans


@njit(cache=True)
def run_qlearning(zb, sb, valid, h, Q, eps0, eps_min, decay, lr, S_true, C_true, N_true, Smean_true, isi_true,
                  cp_fix, th, bits_tab, gsize, n_fft, cp_opts, ser_tar, seed):
    """Tabular Q-learning link adaptation of the reference VLC-MBSN scheme: state = (quantised DC gain,
    quantised mean SNR), action = one constellation for all subcarriers, reward = log2(M)(1-SER) when the
    target is met and -SER otherwise, epsilon-greedy exploration, update Q <- (1-lr) Q + lr r."""
    np.random.seed(seed)
    T = zb.shape[0]
    G = S_true.shape[1]
    nA = bits_tab.shape[0]
    se = np.full(T, np.nan); va = np.full(T, np.nan); vm = np.full(T, np.nan)
    act = np.zeros(T, np.int64)
    eps = eps0
    for t in range(T):
        td = t - h
        if td >= 0 and valid[td]:
            a = act[td]
            b = bits_tab[a]
            ok = 1
            ssum = 0.0
            nsym = 0.0
            for g in range(G):
                gt = S_true[t, g] / (N_true[t] + C_true[t, g] + 2.0 * isi_true[t, cp_fix] * Smean_true[t])
                if b > 0:
                    if b == 1:
                        sr = 0.5 * _erfc(np.sqrt(gt))
                    else:
                        M = 2.0 ** b
                        p = 2.0 * (1.0 - 1.0 / np.sqrt(M)) * 0.5 * _erfc(np.sqrt(1.5 * gt / (M - 1.0)))
                        sr = 1.0 - (1.0 - p) ** 2
                    if sr > ser_tar:
                        ok = 0
                    ssum += sr * gsize[g]
                    nsym += gsize[g]
            mser = ssum / nsym if nsym > 0 else 0.0
            r = b * (1.0 - mser) if mser <= ser_tar else -mser
            Q[zb[td], sb[td], a] = (1.0 - lr) * Q[zb[td], sb[td], a] + lr * r
            tot = 0.0
            for g in range(G):
                tot += b * gsize[g]
            se[td] = tot / (n_fft + cp_opts[cp_fix])
            va[td] = 1.0 - ok
            vm[td] = 1.0 if mser > ser_tar else 0.0
        if not valid[t]:
            continue
        if t % 10 == 0:
            eps = max(eps * decay, eps_min)
        if np.random.random() < eps:
            act[t] = np.random.randint(nA)
        else:
            best = 0
            for a in range(nA):
                if Q[zb[t], sb[t], a] > Q[zb[t], sb[t], best]:
                    best = a
            act[t] = best
    return se, va, vm, Q, eps
