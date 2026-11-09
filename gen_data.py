"""Generate the channel/IMU corpus for all clips, rooms and occupancy settings."""
import os, sys, json, numpy as np
from multiprocessing import Pool
import rooms, simulate as S

BVH = os.environ.get('BVH_DIR', './bvh')
OUT = os.environ.get('DATA_DIR', './data')

TRAIN = ['16_%02d' % i for i in list(range(11, 35)) + [47, 58]] + ['07_%02d' % i for i in range(1, 13)]
VAL = ['08_%02d' % i for i in range(1, 12)]
TEST_WALK = ['02_01', '02_02', '05_01', '06_01', '12_01', '12_02', '12_03', '26_01', '27_01', '10_04'] + \
            ['17_%02d' % i for i in range(1, 10)] + ['15_01', '15_03', '15_09', '15_14']
TEST_CARE = ['26_09', '26_10', '26_11', '02_06', '15_06', '15_07', '13_10', '13_12', '14_07', '14_08', '14_09',
             '13_01', '13_02', '13_03', '13_04', '13_05', '13_06', '14_27', '14_28', '14_29', '14_31', '14_32', '15_10']
TEST_RUN = ['09_01', '09_02', '09_03', '09_04', '09_05', '09_06', '02_03']

JOBS = []
for i, c in enumerate(TRAIN):
    for rep in range(6):
        JOBS.append(('train', c, 'ICU', 2, 1000 + 10 * i + rep))
for i, c in enumerate(VAL):
    JOBS.append(('val', c, 'ICU', 2, 2000 + i))
for i, c in enumerate(TEST_WALK):
    JOBS.append(('S0_walk_icu', c, 'ICU', 2, 3000 + i))
    JOBS.append(('S3_walk_ftpr', c, 'FTPR', 1, 4000 + i))
    JOBS.append(('S4_walk_icu_crowded', c, 'ICU', 6, 5000 + i))
for i, c in enumerate(TEST_CARE):
    JOBS.append(('S1_care_icu', c, 'ICU', 2, 6000 + i))
for i, c in enumerate(TEST_RUN):
    JOBS.append(('S2_run_icu', c, 'ICU', 2, 7000 + i))


def run(job):
    split, clip, room, nst, seed = job
    fn = f'{OUT}/{split}__{clip}__{seed}.npz'
    if os.path.exists(fn):
        return fn
    r = S.simulate(f'{BVH}/{clip}.bvh', rooms.icu() if room == 'ICU' else rooms.ftpr(), nst, seed)
    if r is None:
        return None
    np.savez_compressed(fn, **r)
    return fn


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(os.environ.get('NPROC', 2))) as p:
        out = p.map(run, JOBS, chunksize=1)
    print(sum(o is not None for o in out), 'of', len(JOBS), 'simulated')
