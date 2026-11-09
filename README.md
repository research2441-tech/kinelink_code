# KINELINK

Kinematics-informed predictive link adaptation with online violation control for bidirectional optical
medical body sensor networks (visible-light downlink, infrared uplink).

All files sit in one folder. `run_all.sh` regenerates the corpus, trains the forecasters, runs every
experiment and draws every figure. Everything runs on a CPU; with two cores the full pipeline takes a few hours.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Motion capture | `download_cmu.py`, `bvh.py` | 102 CMU clips, forward kinematics at 100 Hz |
| Rooms and ray tracing | `rooms.py`, `raytrace.py`, `simulate.py`, `gen_data.py` | ICU / FTPR channels (LOS, first-order diffuse, integrating-sphere tail), IMU signals, 4 nodes, 2 links |
| DCO-OFDM link model | `phy.py` | clipping (Bussgang), noise, group SINR, M-QAM SER, bit loading |
| Forecasters | `features.py`, `models.py`, `train_models.py`, `eval_forecast.py` | LSTM (+IMU), GRU, LRU (+IMU), ridge AR |
| Online controllers | `la_core.py` (Numba), `volatility.py`, `run_la.py`, `jobs.py` | KINELINK, ablations, OLLA, Q-learning, split-conformal, fixed, oracle |
| Analysis | `analyze.py`, `make_results.py`, `corpus_stats.py`, `timing.py` | seed statistics, Wilcoxon + Holm tests, figures |
| Figures | `fig_arch.py`, `fig_channel.py`, `make_results.py` | Figs. 1-8 |

## Splits (subject-disjoint)

* train: CMU subjects 07 and 16 (walking), six random placements per clip in the ICU
* validation: subject 08
* S0 walk/ICU, S1 care activities/ICU, S2 jog/ICU, S3 walk/unseen FTPR, S4 walk/ICU with six staff:
  subjects 02, 05, 06, 09, 10, 12, 13, 14, 15, 17, 26, 27

## Key settings

Downlink 50 MHz / N = 128, uplink 20 MHz / N = 64, SER target 1e-3, frame-violation target alpha = 1 %,
horizons D = 1, 2, 5 frames (10 ms each), tracker step eta = 0.05.

Motion data: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), BVH conversion from
github.com/una-dinosauria/cmu-mocap.
