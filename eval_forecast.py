"""Forecast accuracy on held-out subjects/scenarios: RMSE of DC-gain change (dB) and delay spread (ns)."""
import os, pickle, json
import numpy as np
import features as F, models as Mo

MODEL_DIR = os.environ.get('MODEL_DIR', './models_out')
OUT = os.environ.get('RES_DIR', './results')
SPLITS = ['S0_walk_icu', 'S1_care_icu', 'S2_run_icu', 'S3_walk_ftpr', 'S4_walk_icu_crowded']
KINDS = ['persist', 'ar', 'lstm', 'gru', 'lru', 'lstm_imu', 'lru_imu']


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for split in SPLITS:
        st1 = F.streams(split, seed=500, imu=True)
        st0 = F.streams(split, seed=500, imu=False)
        for kind in KINDS:
            seeds = [0] if kind in ('persist', 'ar') else [0, 1, 2]
            for sd in seeds:
                if kind == 'ar':
                    ar = pickle.load(open(f'{MODEL_DIR}/ar.pkl', 'rb'))
                elif kind != 'persist':
                    d = pickle.load(open(f'{MODEL_DIR}/{kind}_s{sd}.pkl', 'rb'))
                err = {}
                for s1, s0 in zip(st1, st0):
                    if kind == 'persist':
                        pr = np.zeros_like(s1['Y']); pr[:, 3:] = s1['U'][:, 1:2]
                    elif kind == 'ar':
                        pr = ar.predict(s0['U'])
                    else:
                        pr = Mo.predict(d['kind'], d['p'], (s1 if d['imu'] else s0)['U'])
                    Y = s1['Y']
                    for k, h in enumerate(F.HORIZONS):
                        m = ~np.isnan(Y[:, k])
                        ez = (pr[m, k] - Y[m, k])
                        et = 10 * (np.exp(pr[m, 3 + k]) - np.exp(Y[m, 3 + k]))     # ns
                        key = (s1['link'], s1['node'], h)
                        err.setdefault(key, [[], []])
                        err[key][0].append(ez); err[key][1].append(et)
                for key, (ez, et) in err.items():
                    ez = np.concatenate(ez); et = np.concatenate(et)
                    res[(split, kind, sd) + key] = (float(np.sqrt(np.mean(ez ** 2))), float(np.sqrt(np.mean(et ** 2))),
                                                    float(np.mean(np.abs(ez))), len(ez))
        print('done', split, flush=True)
    pickle.dump(res, open(f'{OUT}/forecast_rmse.pkl', 'wb'))


if __name__ == '__main__':
    main()
