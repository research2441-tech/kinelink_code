"""Run the experiment grid with a small worker pool."""
import subprocess, sys, itertools
from multiprocessing import Pool
SC = ['S0_walk_icu', 'S1_care_icu', 'S2_run_icu', 'S3_walk_ftpr', 'S4_walk_icu_crowded']
jobs = []
for h in (5, 2):
    for sd in (0, 1, 2):
        for sc in SC:
            jobs.append((sc, sd, h, 0.01, 0.05))
for sd in (0, 1, 2):
    jobs.append(('S0_walk_icu', sd, 1, 0.01, 0.05))
for a in (0.005, 0.02, 0.05):
    jobs.append(('S0_walk_icu', 0, 5, a, 0.05))
    jobs.append(('S1_care_icu', 0, 5, a, 0.05))
for e in (0.01, 0.02, 0.1, 0.2):
    jobs.append(('S0_walk_icu', 0, 5, 0.01, e))


def run(j):
    sc, sd, h, a, e = j
    r = subprocess.run(['python3', 'run_la.py', sc, str(sd), str(h), str(a), str(e)], capture_output=True, text=True)
    open('jobs.log', 'a').write(r.stdout + r.stderr[-2000:] + '\n')
    return j


if __name__ == '__main__':
    with Pool(2) as p:
        for j in p.imap_unordered(run, jobs):
            print('done', j, flush=True)
