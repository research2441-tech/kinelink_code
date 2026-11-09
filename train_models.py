"""Train all forecasters (proposed LRU with inertial input and baselines) over several seeds."""
import os, pickle, time, sys
import numpy as np
import features as F, models as Mo

OUT = os.environ.get('MODEL_DIR', './models_out')
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
KINDS = os.environ.get('KINDS', 'lru_imu,lru,lstm,gru,lstm_imu').split(',')
EPOCHS = int(os.environ.get('EPOCHS', 40))


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(f'{OUT}/train.log', 'a')
    say = lambda m: (log.write(m + '\n'), log.flush())
    cache = {}
    for imu in (True, False):
        tr = F.streams('train', seed=11, imu=imu)
        va = F.streams('val', seed=12, imu=imu)
        cache[imu] = (F.windows(tr), F.windows(va, hop=128))
        if not imu:
            ar = Mo.RidgeAR().fit(tr)
            pickle.dump(ar, open(f'{OUT}/ar.pkl', 'wb'))
    say(f'train windows {cache[True][0][0].shape}, val windows {cache[True][1][0].shape}')
    for kind in KINDS:
        base, imu = kind.replace('_imu', ''), kind.endswith('_imu')
        for s in SEEDS:
            fn = f'{OUT}/{kind}_s{s}.pkl'
            if os.path.exists(fn):
                continue
            t0 = time.time()
            p, hist = Mo.train(base, cache[imu][0], cache[imu][1], seed=s, epochs=EPOCHS, log=say)
            say(f'{kind} seed{s} done in {time.time() - t0:.1f}s params {Mo.n_params(p)} best val {min(h[1] for h in hist):.4f}')
            pickle.dump({'p': p, 'hist': hist, 'kind': base, 'imu': imu, 'time': time.time() - t0}, open(fn, 'wb'))


if __name__ == '__main__':
    main()
