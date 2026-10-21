# IMU Synchronisation Check

`imu_sync_check.py` checks whether several IMUs logged in **one CSV file** are time-synchronised.
It finds the motion spikes that all IMUs see, measures the **lag (delay)** of each IMU against a
reference IMU, and produces figures, log files and a SYNCHRONISED / NOT SYNCHRONISED verdict.

## 1. Test procedure (how the data should be recorded)

1. Bind all IMUs together so they move as **one rigid body**.
2. Start logging and keep everything still for a few seconds.
3. Move the bundle sharply (shake / tap / swing), then **stop** for a few seconds.
4. Repeat step 3 several times (the example file has 7 repetitions).

Every sharp movement creates a spike that all IMUs must record at the same time. If one IMU is
delayed, its spikes appear later than the others.

You do **not** need the `cv_*` columns; the script ignores them.

## 2. Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`

```bash
pip install numpy pandas scipy matplotlib
```

## 3. Usage

```bash
python imu_sync_check.py imu_imu_sync_20261021-190651.csv
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--outdir DIR` | `sync_results` | Output folder |
| `--ref N` | `1` | Reference IMU. All lags are measured against it |
| `--signal gyro\|accel` | `gyro` | Signal used to find spikes (gyro magnitude, or accel magnitude minus gravity) |
| `--prominence P` | auto | Absolute spike prominence. Auto = `--prom-frac` x median IMU maximum |
| `--prom-frac F` | `0.25` | Fraction used for the automatic prominence |
| `--min-dist S` | `0.2` | Minimum time between two spikes in one IMU (s) |
| `--match-window S` | `0.2` | Max time difference for the same spike in two IMUs (s) |
| `--cycle-gap S` | `2.0` | A gap longer than this starts a new move/stop cycle (s) |
| `--max-lag S` | `1.0` | Max lag searched by whole-record cross-correlation (s) |
| `--tol-ms MS` | one sample | Tolerance for the verdict. Default is one sample period (50 ms at 20 Hz) |
| `--prefix TXT` | `ESP32_IMU_` | Column prefix. Columns are `<prefix><n>_ax, ay, az, gx, gy, gz` |
| `--time-col NAME` | `sys_time` | Time column in seconds |
| `--selftest` | | Runs a synthetic test with known lags to prove the method works |

Exit code is `0` if SYNCHRONISED, `2` otherwise.

## 4. Method

1. **Signal.** For each IMU, the gyro magnitude `sqrt(gx²+gy²+gz²)` is computed. It does not depend
   on how the sensor is mounted, and gravity does not affect it.
2. **Spike detection.** `scipy.signal.find_peaks` with an automatic prominence threshold.
3. **Spike matching.** Each spike of the reference IMU is matched with the nearest unused spike in
   every other IMU (within `--match-window`). Only spikes found in **all** IMUs are used.
4. **Spike timing.** The time of a spike is the centroid of the part above half height. This is
   used instead of the raw peak sample because real spikes are often flat-topped (about 3 samples
   wide); on such spikes the highest sample can jump by 1-2 samples because of tiny amplitude
   noise, which would give false lags of 50-100 ms.
5. **Lag** = spike time of IMU *n* minus spike time of the reference IMU, in ms.
   **Positive lag means that IMU sees the spike later than the reference.**
6. **Cross-check.** Normalised cross-correlation of the whole recording, and of every move/stop
   cycle, gives a second lag estimate with sub-sample interpolation.
7. **Verdict.** SYNCHRONISED if, for every IMU, the median spike lag and the cross-correlation lag
   are within the tolerance and at least 90 % of spikes are within the tolerance.

## 5. Output files (in `--outdir`)

| File | Content |
|---|---|
| `fig1_overview.png` | All IMUs overlaid: gyro magnitude, dominant gyro axis, dominant accel axis |
| `fig2_zoom_cycles.png` | Zoom on every move/stop cycle, with the actual samples shown as dots |
| `fig3_lag_analysis.png` | Lag of each spike, lag distribution per IMU, cross-correlation curves, spread per spike |
| `sync_log.txt` | Full human-readable log: data checks, per-spike lags, per-IMU summary, verdict |
| `spike_lags.csv` | One row per spike: peak sample, time, lag (samples and ms) for every IMU, spread |
| `lag_summary.csv` | Per IMU: mean, median, std, min, max lag, cross-correlation lag, drift |
| `xcorr_lags.csv` | Cross-correlation lag for the whole record and for each cycle |
| `pairwise_lag_ms.csv` | Median lag between every pair of IMUs |
| `summary.json` | Verdict and key numbers for use in other tools |

## 6. How to read the results

- **Verdict + max spread** in `sync_log.txt` is the quick answer.
- **fig1** should show the five traces lying on top of each other. A delayed IMU shows as a
  separate trace shifted to the right.
- **fig3 top-left / top-right**: points should sit inside the grey band (+/-1 sample) and around 0.
- **fig3 bottom-left**: all cross-correlation peaks should be at 0 ms.
- **Pairwise lag** (`pairwise_lag_ms.csv`) shows which pair of IMUs differs most, independent of the
  chosen reference.

## 7. Limitations

- **Resolution is one sample.** The file is logged at 20 Hz, so one sample = 50 ms. Lag values
  below about 10-15 ms are interpolated estimates and should be treated as noise. The test can
  show that IMUs agree within 1 sample, but not finer. For finer checks, log at a higher rate.
- **What is being checked.** The timestamps in the file are perfectly uniform (no jitter at all),
  which suggests they are assigned by the logging program when each row is written, not by each
  sensor's own clock. The test therefore checks that the *logged rows* are aligned. It cannot see
  a delay that is identical for all IMUs, and it cannot prove the sensors' internal clocks agree.
- **Drift.** The drift column (ms per minute) is only indicative on a recording of under a
  minute. For a real clock-drift test, record 10 minutes or more with periodic shakes.
- **Stationary periods** contain no information. Lag can only be measured at spikes.


