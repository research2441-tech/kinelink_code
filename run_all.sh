#!/usr/bin/env bash
# Reproduces every table and figure of the KINELINK study (CPU only).
set -e
python3 download_cmu.py            # 102 BVH clips from the CMU database (public)
python3 gen_data.py                # ray-traced channels + IMU for all splits -> ./data
python3 corpus_stats.py            # Table IV statistics -> ./results/corpus_stats.json
python3 train_models.py            # forecasters, seeds 0-2 -> ./models_out
python3 eval_forecast.py           # Table V / Fig. 4 forecast errors
python3 jobs.py                    # all link-adaptation runs (scenarios, horizons, alpha, eta)
python3 timing.py                  # Table XI cost measurements
python3 make_results.py            # results dictionary and Figs. 4-8
python3 fig_arch.py && python3 fig_channel.py   # Figs. 1-3
