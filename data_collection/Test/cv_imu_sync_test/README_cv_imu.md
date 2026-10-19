# CV vs IMU Synchronisation Check

`cv_imu_sync_check.py` checks whether a computer-vision (CV) landmark (for example `cv_0`) and an
IMU (for example IMU 1) are time-synchronised. It finds the spikes both see during head motion,
measures the **lag (delay)** of the CV data relative to the IMU, cross-checks the lag two more ways,
and produces figures, log files and a SYNCHRONISED / NOT SYNCHRONISED verdict.

## 1. Test procedure (how the data should be recorded)

1. Keep the head still for a few seconds.
2. Move the head back and forth (a burst of several swings), then **stop** for a few seconds.
3. Repeat step 2 several times (the example file has 3 bursts).

Only the CV landmark on the head (`cv_0`) and the IMU on the head (IMU 1) need to move. The other
`cv_*` points and IMUs are ignored.

## 2. Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`

```bash
pip install numpy pandas scipy matplotlib
```

## 3. Usage

```bash
python3 cv_imu_sync_check.py test_20261019-154647.csv
python3 cv_imu_sync_check.py --selftest        # proves the method on synthetic data with known delays
```

| Option | Default | Meaning |
|---|---|---|
| `--outdir DIR` | `cv_imu_results` | Output folder |
| `--cv-point N` | `0` | CV landmark number (columns `cv_<N>_x, y, z`) |
| `--imu N` | `1` | IMU number (columns `<prefix><N>_ax ... gz`) |
| `--imu-prefix TXT` | `ESP32_IMU_` | IMU column prefix |
| `--time-col NAME` | `sys_time` | Time column in seconds |
| `--smooth N` | `3` | Moving-average length in samples, applied to both signals (1 = off) |
| `--no-stale-fix` | off | Do not interpolate repeated CV rows |
| `--prom-frac F` | `0.25` | Spike prominence = F x signal maximum |
| `--min-dist S` | `0.2` | Minimum time between two spikes (s) |
| `--match-window S` | `0.25` | Window around the expected CV spike for matching (s) |
| `--burst-gap S` | `2.0` | A gap longer than this starts a new motion burst (s) |
| `--max-lag S` | `1.0` | Max lag searched, whole record (s) |
| `--max-lag-burst S` | `0.5` | Max lag searched, per burst (s) |
| `--hp-hz`, `--lp-hz` | `0.3`, `4.0` | Angle check: high-pass (drift) and low-pass (noise) cut-offs in Hz |
| `--tol-ms MS` | one sample | Tolerance for the verdict (50 ms at 20 Hz) |

Exit code is `0` if SYNCHRONISED, `2` otherwise.

## 4. Method

CV gives **position**; the IMU gives **angular rate**. They cannot be compared directly, so the
script builds like-for-like signals:

1. **CV speed** = length of the derivative of `cv_<N>_x, y, z` (central difference, adds no delay).
2. **IMU activity** = gyro magnitude `sqrt(gx² + gy² + gz²)`. Both signals peak in the middle of each
   swing. The accelerometer is deliberately **not** used: acceleration peaks at the turn-around of a
   swing, a quarter cycle away from the speed peak, so it would give a wrong, phase-shifted lag.
3. **Stale frames.** A CV row identical to the previous row (x, y and z) is a repeated camera frame.
   It is treated as missing and interpolated, because otherwise it fakes a stop followed by a jump.
4. **Spike detection** with `scipy.signal.find_peaks` in both signals.
5. **Matching.** The whole-record cross-correlation gives a coarse lag; each IMU spike is matched to the
   nearest CV spike expected at that lag (within `--match-window`).
6. **Spike timing.** The time of a spike is the centroid of the part above half height (robust to
   flat-topped spikes). **Lag = CV spike time - IMU spike time**; **positive means CV is delayed.**
   A bootstrap gives a 95 % confidence interval of the median lag.
7. **Cross-correlation** of the two signals over the whole record and over each motion burst
   (sub-sample interpolated).
8. **Independent angle check.** The gyro is integrated (trapezoid rule) into a head angle and
   correlated with the CV position (all gyro/CV axis pairs tried, best reported). This does not
   differentiate the noisy CV data, so it does not share the weaknesses of the spike method. Both
   signals go through the same zero-phase filters, which add no delay.
9. **Verdict.** The best estimate is the median of the three lag estimates (spike median, speed
   cross-correlation, angle cross-correlation). SYNCHRONISED if its size is within the tolerance;
   otherwise NOT SYNCHRONISED, with a note on whether the delay is stable across bursts and a
   suggested shift in samples.

## 5. Output files (in `--outdir`)

| File | Content |
|---|---|
| `fig1_overview.png` | CV position, IMU gyro, and the scaled activity signals with detected spikes |
| `fig2_zoom_bursts.png` | Zoom on every motion burst with IMU and CV spikes marked |
| `fig3_lag_analysis.png` | Lag of every spike, lag histogram, cross-correlation curves, before/after shift |
| `fig4_angle_check.png` | Integrated gyro angle vs CV position, as logged and after shifting, plus its cross-correlation |
| `sync_log.txt` | Full log: data checks, per-spike lags, summary, per-burst table, angle check, verdict |
| `spike_lags.csv` | One row per matched spike: IMU time, CV time, lag, amplitudes |
| `burst_summary.csv` | Per burst: spike median lag and cross-correlation lag |
| `xcorr_curve.csv` | Whole-record cross-correlation curve |
| `signals.csv` | The two activity signals, and the CV signal shifted back by the measured lag |
| `summary.json` | Verdict, all lag estimates, best estimate, angle-check numbers |

## 6. How to read the results

- **Verdict block** in `sync_log.txt`: the three lag estimates, the best estimate, whether the delay is
  stable, and the suggested shift.
- **fig2**: if the orange (CV) spikes sit to the right of the blue (IMU) spikes in every burst, the CV
  data is delayed.
- **fig4** is the clearest picture: two smooth curves that are visibly shifted as logged and lie on top
  of each other after the shift.
- **Individual spike lags scatter** (here +6 to +259 ms) because CV speed and gyro magnitude are
  different physical quantities and their peak shapes differ. Rely on the median, the cross-correlation
  and the angle check, not on single spikes.

## 7. Limitations

- **Resolution is one sample** (50 ms at 20 Hz). Values finer than about 10-15 ms are interpolated
  estimates. The three methods agree within about 35 ms here.
- **Only the relative delay is measured.** The result says CV is later than the IMU; it cannot say which
  of the two is "wrong". A delay of this kind typically comes from camera exposure and pose-estimation
  latency before the CV value is written to the log.
- **What is checked.** The timestamps are perfectly uniform (no jitter), which suggests they are assigned by
  the logger when each row is written. The test therefore measures how the CV and IMU values are aligned
  *in the log*.
- **Duration.** About 43 s with 3 bursts is enough to see a constant delay, but too short to measure
  clock drift over time.
- Spikes only exist during motion; the still periods carry no timing information.

## 8. Results for `test_20261019-154647.csv`

- 865 rows, 43.2 s, 20 Hz (50.00 ms step, no gaps, no NaNs). 37 CV rows (4.3 %) were repeated frames and
  were interpolated.
- 3 motion bursts (about 4-12 s, 17-31 s, 37-43 s). 35 IMU spikes and 32 CV spikes were found; 32 were
  matched.

| Method | CV lag vs IMU 1 |
|---|---|
| Spike median (95 % CI +105 to +197 ms) | +147 ms |
| Cross-correlation of CV speed and gyro magnitude | +180 ms |
| Integrated gyro angle vs `cv_0_y` position (r = 0.71 -> 0.96) | +166 ms |

- Per burst (angle check): +159, +173, +157 ms. The delay is stable, like a constant latency.
- Verdict: **NOT SYNCHRONISED.** CV is delayed by about **166 ms** (range 147-180 ms) relative to IMU 1,
  much more than the 50 ms (one sample) tolerance.
- Suggested fix: shift the CV data 3 samples (150 ms) earlier, or the IMU data 3 samples later.

Robustness: changing the smoothing (1, 3, 5 samples), the spike prominence (0.15-0.35) and the stale-frame
handling moved the cross-correlation lag only between 176 and 184 ms.

## 9. Self-test

`python3 cv_imu_sync_check.py --selftest` builds synthetic head swings with a known CV delay (+160, 0 and
-60 ms) and 4 % repeated frames, and checks that all three methods recover the delay to within 25 ms
(they currently agree to about 2 ms). It also fixes the sign convention: positive means CV is delayed.
