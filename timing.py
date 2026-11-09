"""Per-frame cost of the online controller (numba) and of the forecaster step (numpy re-implementation)."""
import time, json, pickle, numpy as np
import run_la as R, features as F, models as Mo
out = {}
models = R.load_models(0)
for link in ('dl', 'ul'):
    data, L, _ = R.assemble('S2_run_icu', 0, link, 5, models, 2)
    a = data['wrist']
    I = R.ingredients(a, L, 'lstm_imu')
    row = [R.POLICIES['KINELINK']]
    R.run_rows(I, a, L, 5, row, 0.01, 4)            # compile
    t0 = time.perf_counter(); n = 20
    for _ in range(n):
        R.run_rows(I, a, L, 5, row, 0.01, 4)
    dt = (time.perf_counter() - t0) / n / len(a['H0'])
    out[f'controller_us_{link}'] = dt * 1e6
# forecaster single-step cost in numpy (batch 1), LSTM width 24
p = pickle.load(open('./models_out/lstm_imu_s0.pkl', 'rb'))['p']
P = {k: (np.asarray(v['W']), np.asarray(v['b'])) for k, v in p.items()}
u = np.random.randn(14).astype(np.float32); h = np.zeros(24, np.float32); c = np.zeros(24, np.float32)
gelu = lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
sig = lambda x: 1 / (1 + np.exp(-x))
def step(u, h, c):
    e = gelu(u @ P['enc'][0] + P['enc'][1])
    g = e @ P['Wx'][0] + P['Wx'][1] + h @ P['Wh'][0] + P['Wh'][1]
    i, f, o, z = sig(g[:24]), sig(g[24:48]), sig(g[48:72]), np.tanh(g[72:])
    c = f * c + i * z; h = o * np.tanh(c)
    y = gelu(h @ P['h1'][0] + P['h1'][1]) @ P['h2'][0] + P['h2'][1]
    return y, h, c
for _ in range(100): step(u, h, c)
t0 = time.perf_counter()
for _ in range(20000): step(u, h, c)
out['forecaster_us_numpy'] = (time.perf_counter() - t0) / 20000 * 1e6
out['params'] = {k: Mo.n_params(pickle.load(open(f'./models_out/{k}_s0.pkl', 'rb'))['p']) for k in ('lstm_imu', 'lstm', 'gru', 'lru_imu')}
out['train_s'] = {k: float(np.mean([pickle.load(open(f'./models_out/{k}_s{s}.pkl', 'rb'))['time'] for s in (0, 1, 2)])) for k in ('lstm_imu', 'lstm', 'gru', 'lru_imu')}
json.dump(out, open('./results/timing.json', 'w'), indent=1)
print(out)
