"""Online link-adaptation experiments: all policies, scenarios, horizons and seeds."""
import os, sys, pickle, json, time
import numpy as np
import features as F, models as Mo, phy, la_core as C, volatility as V

MODEL_DIR = os.environ.get('MODEL_DIR', './models_out')
OUT = os.environ.get('RES_DIR', './results')
SER_TAR = 1e-3
TH = phy.snr_threshold(SER_TAR)
BITS = phy.BITS.astype(np.int64)
FORECASTERS = ('lru_imu', 'lstm', 'gru', 'lstm_imu', 'ar', 'persist')

# policy table: name -> (forecaster, score_mode, adapt_q, cp_mode, adapt_tau, split)
POLICIES = {
    'KINELINK':        ('lstm_imu', 2, 1, 0, 1, False),
    'KL uniform margin': ('lstm_imu', 2, 1, 0, 1, False),
    'KL history margin': ('lstm_imu', 2, 1, 0, 1, False),
    'KL coverage score': ('lstm_imu', 0, 1, 0, 1, False),
    'KL w/o IMU':      ('lstm', 2, 1, 0, 1, False),
    'KL fixed CP':     ('lstm_imu', 2, 1, 1, 0, False),
    'KL gain score':   ('lstm_imu', 1, 1, 0, 1, False),
    'KL split CP':     ('lstm_imu', 0, 0, 0, 0, True),
    'LRU-IMU+ACI':     ('lru_imu', 2, 1, 0, 1, False),
    'GRU+ACI':         ('gru', 2, 1, 0, 1, False),
    'AR+ACI':          ('ar', 2, 1, 0, 1, False),
    'Persist+ACI':     ('persist', 2, 1, 0, 1, False),
    'OLLA':            ('persist', 2, 1, 1, 0, False),
    'LSTM+split':      ('lstm', 0, 0, 1, 0, True),
}


VOL = {'KINELINK': True, 'KL history margin': False}     # kinematic (True) or history-only (False) margin scale
KAPPAS = (0.0, 1.0, 2.0, 3.0)
NATS_PER_DB = 2 * np.log(10) / 10                       # optical-gain dB -> electrical SINR nats


def load_models(mseed):
    m = {}
    for k in ('lru_imu', 'lstm', 'gru', 'lstm_imu'):
        d = pickle.load(open(f'{MODEL_DIR}/{k}_s{mseed}.pkl', 'rb'))
        m[k] = d
    m['ar'] = pickle.load(open(f'{MODEL_DIR}/ar.pkl', 'rb'))
    return m


def forecast(models, st_imu, st_noimu, k_h):
    """Per stream forecasts (dz, log-tau) at horizon index k_h for every forecaster."""
    out = {f: [] for f in FORECASTERS}
    for s1, s0 in zip(st_imu, st_noimu):
        T = len(s1['U'])
        for f in FORECASTERS:
            if f == 'persist':
                pr = np.zeros((T, 6)); pr[:, 3 + k_h] = s1['U'][:, 1]
            elif f == 'ar':
                pr = models['ar'].predict(s0['U'])
            else:
                d = models[f]
                pr = Mo.predict(d['kind'], d['p'], (s1 if d['imu'] else s0)['U'])
            out[f].append(np.stack([pr[:, k_h], pr[:, 3 + k_h]], 1))
    return out


def assemble(split, seed, link, h, models, k_h, order_seed=None):
    """Concatenate clips (shuffled order) for the 4 nodes of one link into aligned arrays."""
    st1 = [s for s in F.streams(split, seed=100 + seed, imu=True) if s['link'] == link]
    st0 = [s for s in F.streams(split, seed=100 + seed, imu=False) if s['link'] == link]
    fc = forecast(models, st1, st0, k_h)
    clips = sorted(set(s['file'] for s in st1))
    rng = np.random.default_rng(seed if order_seed is None else order_seed)
    clips = [clips[i] for i in rng.permutation(len(clips))]
    L = phy.Link(link)
    res = {}
    for nd in F.NODES:
        idx = {s['file']: i for i, s in enumerate(st1) if s['node'] == nd}
        parts = {k: [] for k in ('Hf', 'H0', 'isi', 'tau', 'zm', 'taum', 'valid', 'clip', 'U')}
        fparts = {f: [] for f in FORECASTERS}
        for ci, cf in enumerate(clips):
            s = st1[idx[cf]]
            T = len(s['H0'])
            parts['Hf'].append(s['Hf']); parts['H0'].append(s['H0']); parts['isi'].append(s['isi'])
            parts['tau'].append(s['tau'] * 1e-9); parts['zm'].append(s['z_meas']); parts['taum'].append(s['tau_meas'] * 1e-9)
            v = np.zeros(T, bool); v[:T - h] = True
            parts['valid'].append(v); parts['clip'].append(np.full(T, ci)); parts['U'].append(s['U'])
            for f in FORECASTERS:
                fparts[f].append(fc[f][idx[cf]])
        a = {k: np.concatenate(v) for k, v in parts.items()}
        a['fc'] = {f: np.concatenate(v) for f, v in fparts.items()}
        res[nd] = a
    return res, L, clips


def ingredients(a, L, f):
    """Nominal (forecast) and realised SINR ingredients for one node and forecaster."""
    zt = 10 * np.log10(a['H0'])
    meas_scale = 10 ** ((a['zm'] - zt) / 20)
    Hf_meas = a['Hf'] * meas_scale[:, None]
    dz, ltau = a['fc'][f][:, 0], a['fc'][f][:, 1]
    Hhat = Hf_meas * 10 ** (dz / 20)[:, None]
    H0hat = 10 ** ((a['zm'] + dz) / 10)
    tau_hat = np.exp(ltau) * 10e-9
    sK = (L.R * L.K * L.sig) ** 2
    sC = (L.R * L.sig) ** 2 * L.clip_var
    S_hat = sK * Hhat ** 2 * L.led2
    C_hat = sC * Hhat ** 2 * L.led2
    N_hat = L.noise_var(H0hat)
    S_true = sK * a['Hf'] ** 2 * L.led2
    C_true = sC * a['Hf'] ** 2 * L.led2
    N_true = L.noise_var(a['H0'])
    # measured ISI leakage at the decision time (pilot-based tap profile with the same measurement noise)
    isi_meas = a['isi'].astype(np.float64)
    return dict(S_hat=S_hat, C_hat=C_hat, N_hat=N_hat, Smean_hat=S_hat.mean(1), isi_meas=isi_meas,
                tau_meas=a['taum'], tau_hat=tau_hat, S_true=S_true, C_true=C_true, N_true=N_true,
                Smean_true=S_true.mean(1), isi_true=a['isi'].astype(np.float64))


def vol_feats(a, f, h, imu):
    X, E = [], []
    for c in np.unique(a['clip']):
        m = a['clip'] == c
        x, e = V.features(a['U'][m], a['fc'][f][m, 0], a['zm'][m], h, imu)
        X.append(x); E.append(e)
    return X, E


def margin_offset(a, f, h, vm, kappa):
    if vm is None or kappa == 0:
        return np.zeros(len(a['H0']))
    model, imu, sbar = vm
    X, _ = vol_feats(a, f, h, imu)
    sh = model.predict(np.concatenate(X))
    return kappa * NATS_PER_DB * (sh - sbar)


def oracle(a, L):
    sK = (L.R * L.K * L.sig) ** 2
    S = (sK * a['Hf'] ** 2 * L.led2).astype(np.float64)
    Cc = (L.R * L.sig) ** 2 * L.clip_var * a['Hf'] ** 2 * L.led2
    N = L.noise_var(a['H0'])
    best = np.zeros(len(S)); bestL = np.zeros(len(S), int)
    for j, cp in enumerate(L.cp_opts):
        g = S / (N[:, None] + Cc + 2 * a['isi'][:, j:j + 1] * S.mean(1, keepdims=True))
        se = L.se(L.load(g, TH), cp)
        upd = se > best
        best[upd] = se[upd]; bestL[upd] = j
    return best, bestL


def true_sinr(a, L, j):
    sK = (L.R * L.K * L.sig) ** 2
    S = sK * a['Hf'] ** 2 * L.led2
    Cc = (L.R * L.sig) ** 2 * L.clip_var * a['Hf'] ** 2 * L.led2
    return S / (L.noise_var(a['H0'])[:, None] + Cc + 2 * a['isi'][:, j:j + 1] * S.mean(1, keepdims=True))


def train_side(link, h, models, k_h, alpha):
    """Choices learnt on the training split: fixed CP, fixed constellation, Q-table, split-conformal margins."""
    tr, L, _ = assemble('train', 0, link, h, models, k_h, order_seed=0)
    va, _, _ = assemble('val', 0, link, h, models, k_h, order_seed=0)
    # CP maximising oracle SE on training data
    cp_hist = np.zeros(len(L.cp_opts))
    for nd in F.NODES:
        _, bl = oracle(tr[nd], L)
        cp_hist += np.bincount(bl, minlength=len(L.cp_opts))
    cp_fix = int(np.argmax(cp_hist))
    fixed_bits = {}
    for nd in F.NODES:
        g = true_sinr(tr[nd], L, cp_fix)
        best = 0
        for b in BITS[1:]:
            v = (phy.ser(np.array([b]), g) > SER_TAR).any(1).mean()
            if v <= alpha:
                best = b
        fixed_bits[nd] = int(best)
    # split-conformal margins from validation scores at zero margin
    split_q = {}
    for name, (f, sm, aq, cpm, at, sp) in POLICIES.items():
        if not sp:
            continue
        for nd in F.NODES:
            I = ingredients(va[nd], L, f)
            out = run_rows(I, va[nd], L, h, [(f, 0, 0, cpm, 0, False)], alpha, cp_fix, q_init=[0.0], qt_init=[0.0])
            scores = out[5][0][~np.isnan(out[5][0])]
            n = len(scores)
            split_q[(name, nd)] = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
            ts = np.log(va[nd]['taum'][h:]) - np.log(I['tau_hat'][:-h])
            split_q[(name, nd, 'tau')] = float(np.quantile(ts[va[nd]['valid'][:-h]], 1 - alpha))
    # kinematics-conditioned (and history-only) volatility models, kappa chosen on validation
    vol = {}
    for name, imu in VOL.items():
        f = POLICIES[name][0]
        Xs, Es = [], []
        for nd in F.NODES:
            X, E = vol_feats(tr[nd], f, h, imu); Xs += X; Es += E
        model = V.VolModel(imu).fit(Xs, Es)
        sbar = float(np.mean(model.predict(np.concatenate(Xs))))
        best = (-1, 0.0)
        for kp in KAPPAS:
            se_all, v_all = [], []
            for nd in F.NODES:
                I = ingredients(va[nd], L, f)
                off = margin_offset(va[nd], f, h, (model, imu, sbar), kp)
                r = run_rows(I, va[nd], L, h, [POLICIES[name]], alpha, cp_fix, sig=off)
                m = ~np.isnan(r[0][0]); se_all.append(r[0][0][m]); v_all.append(r[1][0][m])
            se_m, v_m = np.mean(np.concatenate(se_all)), np.mean(np.concatenate(v_all))
            if v_m <= alpha + 0.01 and se_m > best[0] + 1e-9:
                best = (se_m, kp)
        vol[name] = (model, imu, sbar, best[1])
    # Q-learning pre-training on training streams
    Qtabs = {}
    for nd in F.NODES:
        Q = np.zeros((60, 40, len(BITS)))
        zb, sb = qstate(tr[nd], L)
        I = ingredients(tr[nd], L, 'persist')
        eps = 1.0
        for ep in range(3):
            _, _, _, Q, eps = C.run_qlearning(zb, sb, tr[nd]['valid'], h, Q, eps, 0.001, 0.99, 0.5,
                                              I['S_true'], I['C_true'], I['N_true'], I['Smean_true'], I['isi_true'],
                                              cp_fix, TH, BITS, L.gsize.astype(np.int64), L.n_fft,
                                              L.cp_opts.astype(np.int64), SER_TAR, 7 + ep)
        Qtabs[nd] = (Q, eps)
    return dict(cp_fix=cp_fix, fixed_bits=fixed_bits, split_q=split_q, Q=Qtabs, vol=vol)


def qstate(a, L):
    zb = np.clip(np.round(a['zm'] + 70).astype(np.int64), 0, 59)
    g = true_sinr(a, L, 0)
    snr_db = 10 * np.log10(np.maximum(g.mean(1), 1e-6))
    snr_meas = snr_db + (a['zm'] - 10 * np.log10(a['H0'])) * 2
    sb = np.clip(np.round((snr_meas + 20) / 2).astype(np.int64), 0, 39)
    return zb, sb


def run_rows(I, a, L, h, rows, alpha, cp_fix, q_init=None, qt_init=None, eta=0.05, eta_tau=0.02, alpha_tau=0.05, sig=None):
    R = len(rows)
    T = len(a['H0'])
    stack = lambda k: np.ascontiguousarray(np.broadcast_to(I[k], (R,) + I[k].shape))
    Is = {k: stack(k) for k in I}
    q0 = np.array(q_init if q_init is not None else [0.5] * R, float)
    qt0 = np.array(qt_init if qt_init is not None else [0.1] * R, float)
    adapt_q = np.array([r[2] for r in rows], np.int64)
    adapt_t = np.array([r[4] for r in rows], np.int64)
    cpm = rows[0][3]
    sm = rows[0][1]
    valid = np.ascontiguousarray(np.broadcast_to(a['valid'], (R, T)))
    return C.run_calibrated(Is['S_hat'], Is['C_hat'], Is['N_hat'], Is['Smean_hat'], Is['isi_meas'], Is['tau_meas'],
                            Is['tau_hat'], Is['S_true'], Is['C_true'], Is['N_true'], Is['Smean_true'], Is['isi_true'],
                            valid, h, alpha, eta, q0, alpha_tau, eta_tau, qt0, adapt_q, adapt_t, cpm, cp_fix, sm,
                            TH, BITS, L.gsize.astype(np.int64), L.n_fft, L.cp_opts.astype(np.int64), SER_TAR, 20.0,
                            np.ascontiguousarray(np.broadcast_to(np.zeros(T) if sig is None else sig, (R, T))))


def evaluate(split, seed, link, h, k_h, models, side, alpha, eta=0.05, policies=None):
    data, L, clips = assemble(split, seed, link, h, models, k_h)
    out = {}
    pols = policies or list(POLICIES)
    for nd in F.NODES:
        a = data[nd]
        o_se, _ = oracle(a, L)
        # oracle SE at the target frame, aligned to decision time t (target t+h)
        se_or = np.full(len(o_se), np.nan); se_or[:len(o_se) - h] = o_se[h:]
        se_or[~a['valid']] = np.nan
        out[(nd, 'Oracle')] = dict(se=se_or, va=np.where(a['valid'], 0.0, np.nan), vm=np.where(a['valid'], 0.0, np.nan),
                                    clip=a['clip'])
        # fixed constellation
        g = true_sinr(a, L, side['cp_fix'])
        b = side['fixed_bits'][nd]
        sr = phy.ser(np.array([b]), g)
        se_f = np.full(len(g), b * L.gsize.sum() / (L.n_fft + L.cp_opts[side['cp_fix']]))
        va_f = (sr > SER_TAR).any(1).astype(float) if b > 0 else np.zeros(len(g))
        vm_f = ((sr * L.gsize).sum(1) / L.gsize.sum() > SER_TAR).astype(float) if b > 0 else np.zeros(len(g))
        sh = lambda x: np.concatenate([x[h:], np.full(h, np.nan)])
        out[(nd, 'Fixed')] = dict(se=np.where(a['valid'], sh(se_f), np.nan), va=np.where(a['valid'], sh(va_f), np.nan),
                                  vm=np.where(a['valid'], sh(vm_f), np.nan), clip=a['clip'])
        # Q-learning (continues online from the pre-trained table)
        Q, eps = side['Q'][nd]
        zb, sb = qstate(a, L)
        Ip = ingredients(a, L, 'persist')
        se_q, va_q, vm_q, _, _ = C.run_qlearning(zb, sb, a['valid'], h, Q.copy(), max(eps, 0.001), 0.001, 0.99, 0.5,
                                                 Ip['S_true'], Ip['C_true'], Ip['N_true'], Ip['Smean_true'], Ip['isi_true'],
                                                 side['cp_fix'], TH, BITS, L.gsize.astype(np.int64), L.n_fft,
                                                 L.cp_opts.astype(np.int64), SER_TAR, 1000 + seed)
        out[(nd, 'Q-learning')] = dict(se=se_q, va=va_q, vm=vm_q, clip=a['clip'])
        cacheI = {}
        for name in pols:
            f, sm, aq, cpm, at, sp = POLICIES[name]
            if f not in cacheI:
                cacheI[f] = ingredients(a, L, f)
            q_init = [side['split_q'][(name, nd)]] if sp else [0.5]
            qt_init = [side['split_q'][(name, nd, 'tau')]] if sp else [0.1]
            off = None
            if name in VOL:
                model, imu, sbar, kp = side['vol'][name]
                off = margin_offset(a, f, h, (model, imu, sbar), kp)
            r = run_rows(cacheI[f], a, L, h, [(f, sm, aq, cpm, at, sp)], alpha, side['cp_fix'], q_init, qt_init, eta=eta, sig=off)
            out[(nd, name)] = dict(se=r[0][0], va=r[1][0], vm=r[2][0], q=r[3][0], L=r[4][0], score=r[5][0], clip=a['clip'])
    return out, L, clips


def summarise(out):
    rows = {}
    for (nd, name), d in out.items():
        m = ~np.isnan(d['se'])
        rows[(nd, name)] = dict(se=float(np.mean(d['se'][m])), viol=float(np.nanmean(d['va'][m])),
                                viol_mean=float(np.nanmean(d['vm'][m])), n=int(m.sum()))
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    split = sys.argv[1]; seed = int(sys.argv[2]); h = int(sys.argv[3]); alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    eta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    k_h = F.HORIZONS.index(h)
    models = load_models(seed % 3)
    for link in F.LINKS:
        fn_side = f'{OUT}/side_{link}_h{h}_a{alpha}_m{seed % 3}.pkl'
        tag = f'{split}_{link}_h{h}_a{alpha}_e{eta}_s{seed}'
        if os.path.exists(f'{OUT}/la_{tag}.pkl'):
            continue
        if os.path.exists(fn_side):
            side = pickle.load(open(fn_side, 'rb'))
        else:
            side = train_side(link, h, models, k_h, alpha)
            pickle.dump(side, open(fn_side, 'wb'))
        t0 = time.time()
        out, L, clips = evaluate(split, seed, link, h, k_h, models, side, alpha, eta=eta)
        slim = {k: {kk: (vv.astype(np.float32) if vv.dtype.kind == 'f' else vv) for kk, vv in d.items()} for k, d in out.items()}
        pickle.dump({'out': slim, 'clips': clips}, open(f'{OUT}/la_{tag}.pkl', 'wb'))
        summ = summarise(out)
        print(split, link, 'h', h, 'seed', seed, f'{time.time() - t0:.1f}s')
        for name in ['Oracle', 'Fixed', 'Q-learning'] + list(POLICIES):
            se = np.mean([summ[(nd, name)]['se'] for nd in F.NODES])
            vi = np.mean([summ[(nd, name)]['viol'] for nd in F.NODES])
            print(f'   {name:16s} SE {se:.3f} viol {vi:.4f}')
