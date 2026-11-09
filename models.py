"""Sequence forecasters in JAX: diagonal linear recurrent unit (proposed), GRU, LSTM; ridge AR baseline."""
import numpy as np
import jax, jax.numpy as jnp
import optax

N_OUT = 6


def _dense(key, n_in, n_out, scale=1.0):
    return {'W': jax.random.normal(key, (n_in, n_out)) * scale / np.sqrt(n_in), 'b': jnp.zeros(n_out)}


def init(kind, key, n_in, d=24, n=32):
    ks = jax.random.split(key, 8)
    p = {'enc': _dense(ks[0], n_in, d), 'h1': _dense(ks[1], d, 32), 'h2': _dense(ks[2], 32, N_OUT, 0.1)}
    if kind == 'lru':
        r = jax.random.uniform(ks[3], (n,), minval=0.0, maxval=1.0)
        mod = jnp.sqrt(r * (0.999 ** 2 - 0.90 ** 2) + 0.90 ** 2)            # |lambda| in [0.90, 0.999]
        p['nu'] = jnp.log(-jnp.log(mod))
        p['theta'] = jax.random.uniform(ks[4], (n,), minval=0.0, maxval=0.3)
        p['B_re'] = jax.random.normal(ks[5], (d, n)) / np.sqrt(2 * d)
        p['B_im'] = jax.random.normal(ks[6], (d, n)) / np.sqrt(2 * d)
        p['C_re'] = jax.random.normal(ks[7], (n, d)) / np.sqrt(n)
        p['C_im'] = jax.random.normal(jax.random.fold_in(ks[7], 1), (n, d)) / np.sqrt(n)
        p['D'] = jnp.zeros(d)
    elif kind == 'gru':
        p['Wx'] = _dense(ks[3], d, 3 * d); p['Wh'] = _dense(ks[4], d, 3 * d)
    elif kind == 'lstm':
        p['Wx'] = _dense(ks[3], d, 4 * d); p['Wh'] = _dense(ks[4], d, 4 * d)
        p['Wx']['b'] = p['Wx']['b'].at[d:2 * d].set(1.0)                   # forget-gate bias
    return p


def _lin(q, x):
    return x @ q['W'] + q['b']


def apply(kind, p, U):
    """U: (B, T, n_in) -> (B, T, N_OUT). Causal; state starts at zero at t=0."""
    e = jax.nn.gelu(_lin(p['enc'], U))
    Bsz, T, d = e.shape
    if kind == 'lru':
        lam = jnp.exp(-jnp.exp(p['nu']) + 1j * p['theta'])
        gam = jnp.sqrt(1 - jnp.abs(lam) ** 2)
        Bc = (p['B_re'] + 1j * p['B_im']) * gam[None, :]
        Cc = p['C_re'] + 1j * p['C_im']
        bu = jnp.einsum('btd,dn->btn', e.astype(jnp.complex64), Bc)

        def step(x, b):
            x = lam * x + b
            return x, x
        _, xs = jax.lax.scan(step, jnp.zeros((Bsz, lam.shape[0]), jnp.complex64), jnp.swapaxes(bu, 0, 1))
        xs = jnp.swapaxes(xs, 0, 1)
        h = jnp.real(jnp.einsum('btn,nd->btd', xs, Cc)) + p['D'] * e
    elif kind == 'gru':
        def step(hh, x):
            gx = _lin(p['Wx'], x); gh = _lin(p['Wh'], hh)
            r = jax.nn.sigmoid(gx[:, :d] + gh[:, :d]); z = jax.nn.sigmoid(gx[:, d:2 * d] + gh[:, d:2 * d])
            nn_ = jnp.tanh(gx[:, 2 * d:] + r * gh[:, 2 * d:])
            hh = (1 - z) * nn_ + z * hh
            return hh, hh
        _, hs = jax.lax.scan(step, jnp.zeros((Bsz, d)), jnp.swapaxes(e, 0, 1))
        h = jnp.swapaxes(hs, 0, 1)
    elif kind == 'lstm':
        def step(c, x):
            hh, cc = c
            g = _lin(p['Wx'], x) + _lin(p['Wh'], hh)
            i, f, o, u = (jax.nn.sigmoid(g[:, :d]), jax.nn.sigmoid(g[:, d:2 * d]),
                          jax.nn.sigmoid(g[:, 2 * d:3 * d]), jnp.tanh(g[:, 3 * d:]))
            cc = f * cc + i * u; hh = o * jnp.tanh(cc)
            return (hh, cc), hh
        _, hs = jax.lax.scan(step, (jnp.zeros((Bsz, d)), jnp.zeros((Bsz, d))), jnp.swapaxes(e, 0, 1))
        h = jnp.swapaxes(hs, 0, 1)
    z = jax.nn.gelu(_lin(p['h1'], h))
    return _lin(p['h2'], z)


W_LOSS = jnp.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])


def loss_fn(kind, p, U, Y, M):
    pred = apply(kind, p, U)
    return (((pred - Y) ** 2) * M * W_LOSS).sum() / jnp.maximum(M.sum(), 1.0)


def train(kind, data_tr, data_va, seed=0, epochs=40, bs=64, lr=3e-3, log=None):
    U, Y, M = data_tr
    key = jax.random.PRNGKey(seed)
    p = init(kind, key, U.shape[-1])
    steps = epochs * int(np.ceil(len(U) / bs))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-4))
    st = opt.init(p)

    @jax.jit
    def upd(p, st, u, y, m):
        l, g = jax.value_and_grad(lambda q: loss_fn(kind, q, u, y, m))(p)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l

    val = jax.jit(lambda q, u, y, m: loss_fn(kind, q, u, y, m))
    rng = np.random.default_rng(seed)
    best, best_p, hist = np.inf, p, []
    for ep in range(epochs):
        idx = rng.permutation(len(U))
        tl = []
        for a in range(0, len(U), bs):
            b = idx[a:a + bs]
            p, st, l = upd(p, st, U[b], Y[b], M[b])
            tl.append(float(l))
        vl = float(np.mean([val(p, data_va[0][a:a + 256], data_va[1][a:a + 256], data_va[2][a:a + 256])
                            for a in range(0, len(data_va[0]), 256)]))
        hist.append((np.mean(tl), vl))
        if vl < best:
            best, best_p = vl, jax.tree_util.tree_map(lambda x: x, p)
        if log:
            log(f'{kind} seed{seed} ep{ep} train {np.mean(tl):.4f} val {vl:.4f}')
    return best_p, hist


_JIT = {}


def predict(kind, p, U):
    """Causal prediction over a full sequence; inputs are zero-padded to a 512-multiple to limit re-compilation."""
    if kind not in _JIT:
        _JIT[kind] = jax.jit(lambda q, u: apply(kind, q, u))
    T = len(U)
    Tp = int(np.ceil(T / 512) * 512)
    Up = np.zeros((Tp, U.shape[1]), np.float32); Up[:T] = U
    return np.asarray(_JIT[kind](p, Up[None]))[0][:T]


def n_params(p):
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(p)))


class RidgeAR:
    """Autoregressive baseline on the last `lag` channel measurements (no inertial input)."""
    def __init__(self, lag=10, lam=1.0):
        self.lag, self.lam = lag, lam

    def _feat(self, U):
        z, t = U[:, 0], U[:, 1]
        T = len(z)
        F = [np.ones(T)]
        for k in range(self.lag):
            F.append(np.concatenate([np.full(k, z[0]), z[:T - k]]) - z)
            F.append(np.concatenate([np.full(k, t[0]), t[:T - k]]))
        return np.stack(F, 1)

    def fit(self, streams):
        X = np.concatenate([self._feat(s['U']) for s in streams])
        Y = np.concatenate([s['Y'] for s in streams])
        ok = ~np.isnan(Y).any(1)
        X, Y = X[ok], Y[ok]
        self.W = np.linalg.solve(X.T @ X + self.lam * np.eye(X.shape[1]), X.T @ Y)
        return self

    def predict(self, U):
        return self._feat(U) @ self.W
