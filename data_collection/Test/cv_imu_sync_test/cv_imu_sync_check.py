#!/usr/bin/env python3
"""
cv_imu_sync_check.py
====================
Check whether a computer-vision (CV) landmark and an IMU are time-synchronised.

Test idea: only the head moves (move -> short stop -> move -> ...). The CV landmark
(e.g. cv_0) and the IMU (e.g. IMU 1) must both see every swing at the same moment.

CV gives POSITION, the IMU gives ANGULAR RATE / ACCELERATION, so they cannot be
compared directly. Both are turned into an "how fast is the head moving" signal:

    CV  : speed of the landmark  = |d(position)/dt|           (derivative of x,y,z)
    IMU : gyro magnitude         = sqrt(gx^2 + gy^2 + gz^2)

Both peak in the middle of every swing, so their spikes should line up. (Accelerometer
magnitude is NOT used: it peaks at the turn-around of each swing, a quarter cycle away
from the speed peak, so it is not a like-for-like partner for CV speed.) The script

  1. finds the spikes in both signals,
  2. matches each IMU spike with the CV spike of the same swing,
  3. measures the lag of the CV spike relative to the IMU spike,
  4. cross-checks with cross-correlation (whole record and every motion burst),
  5. runs an independent check that avoids differentiating the CV data: the gyro is
     integrated into a head angle and correlated with the CV position,
  6. writes figures, log files and CSV/JSON results, and gives a verdict.

Sign convention:  lag > 0  =>  CV sees the motion LATER than the IMU (CV is delayed).

Usage
-----
    python3 cv_imu_sync_check.py test.csv
    python3 cv_imu_sync_check.py test.csv --cv-point 0 --imu 1 --outdir out
    python3 cv_imu_sync_check.py --selftest

See README_cv_imu.md for details.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt  # noqa: E402

IMU_C = "#0072B2"                                    # IMU = blue
CV_C = "#D55E00"                                     # CV  = vermillion
CV_AX = ["#D55E00", "#E69F00", "#CC79A7"]            # warm colours = CV axes
IMU_AX = ["#0072B2", "#56B4E9", "#009E73"]           # cool colours = IMU axes
BURST_C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
log = logging.getLogger("cv_imu_sync")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def setup_logging(outdir: Path | None) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    handlers = [logging.StreamHandler(sys.stdout)]
    if outdir is not None:
        handlers.append(logging.FileHandler(outdir / "sync_log.txt", mode="w", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)


def parabolic(y: np.ndarray, k: int) -> float:
    """Sub-sample position of a peak at index k (3-point parabola fit)."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    den = y0 - 2.0 * y1 + y2
    if den == 0:
        return float(k)
    return k + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))


def spike_centroid(y: np.ndarray, k: int, base: float) -> float:
    """Robust sub-sample spike time (in samples): centroid of the part of the spike
    that is above half height (from resting level `base`) around the peak sample k."""
    half = base + 0.5 * (float(y[k]) - base)
    lo = k
    while lo > 0 and y[lo - 1] >= half:
        lo -= 1
    hi = k
    while hi < len(y) - 1 and y[hi + 1] >= half:
        hi += 1
    idx = np.arange(lo, hi + 1)
    w = y[lo:hi + 1] - half
    if w.sum() <= 0:
        return float(k)
    return float((idx * w).sum() / w.sum())


def xcorr_lag(ref: np.ndarray, other: np.ndarray, dt: float, max_lag_s: float):
    """Lag (s) of `other` relative to `ref` from normalised cross-correlation.
    Positive result => `other` is delayed (arrives later) than `ref`.
    Returns (lag_s, peak_corr, lags_s, corr)."""
    a = ref - ref.mean()
    b = other - other.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    if denom == 0 or len(a) < 3:
        return float("nan"), float("nan"), None, None
    n = len(a)
    full = correlate(b, a, mode="full", method="fft") / denom
    lags = np.arange(-(n - 1), n)
    m = np.abs(lags) <= int(round(max_lag_s / dt))
    lags_m, c_m = lags[m], full[m]
    k = int(np.argmax(c_m))
    pos = parabolic(c_m, k)
    lag_samples = float(np.interp(pos, np.arange(len(lags_m)), lags_m))
    return lag_samples * dt, float(c_m[k]), lags_m * dt, c_m


def smooth(y: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average (adds no delay)."""
    return uniform_filter1d(y, size=n, mode="nearest") if n > 1 else y.copy()


# --------------------------------------------------------------------------
# data loading and signals
# --------------------------------------------------------------------------
def load_data(csv_path: Path, a):
    df = pd.read_csv(csv_path)
    ip = f"{a.imu_prefix}{a.imu}_"
    need = [a.time_col] + [f"cv_{a.cv_point}_{x}" for x in "xyz"] + [ip + k for k in ("ax", "ay", "az", "gx", "gy", "gz")]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {csv_path.name}: {missing}")
    t_abs = df[a.time_col].to_numpy(float)
    return df, t_abs - t_abs[0]


def fix_stale(cv: np.ndarray):
    """A row identical to the previous one in x, y and z is a repeated camera frame.
    Treat it as missing and interpolate, otherwise it fakes a stop followed by a jump."""
    stale = np.r_[False, np.all(np.diff(cv, axis=0) == 0, axis=1)]
    out = cv.copy()
    out[stale] = np.nan
    for j in range(out.shape[1]):
        out[:, j] = pd.Series(out[:, j]).interpolate(limit_direction="both").to_numpy()
    return out, stale


def build_signals(df: pd.DataFrame, t: np.ndarray, a) -> dict:
    dt = float(np.median(np.diff(t)))
    cv_raw = df[[f"cv_{a.cv_point}_{x}" for x in "xyz"]].to_numpy(float)
    cv_fix, stale = (cv_raw, np.zeros(len(cv_raw), bool)) if a.no_stale_fix else fix_stale(cv_raw)
    cv_speed = np.linalg.norm(np.gradient(cv_fix, dt, axis=0), axis=1)
    ip = f"{a.imu_prefix}{a.imu}_"
    imu_act = np.linalg.norm(df[[ip + "gx", ip + "gy", ip + "gz"]].to_numpy(float), axis=1)
    return dict(cv_raw=cv_raw, cv_fix=cv_fix, stale=stale, cv_speed=cv_speed, imu_act=imu_act,
                imu_gyro=df[[ip + "gx", ip + "gy", ip + "gz"]].to_numpy(float),
                cv=smooth(cv_speed, a.smooth), imu=smooth(imu_act, a.smooth))


# --------------------------------------------------------------------------
# core analysis
# --------------------------------------------------------------------------
def shifted(t: np.ndarray, y: np.ndarray, lag_s: float) -> np.ndarray:
    """Move a delayed signal back in time by lag_s: out(t) = y(t + lag_s)."""
    return np.interp(t + lag_s, t, y, left=np.nan, right=np.nan)


def analyse(t: np.ndarray, imu: np.ndarray, cv: np.ndarray, a) -> dict:
    dt = float(np.median(np.diff(t)))
    tol_ms = a.tol_ms if a.tol_ms else dt * 1000.0
    idx_axis = np.arange(len(t))
    res = dict(dt=dt, tol_ms=tol_ms)

    # ---- whole-record cross-correlation ----
    lag_s, pk, lags_s, corr = xcorr_lag(imu, cv, dt, a.max_lag)
    r0 = float(corr[int(np.argmin(np.abs(lags_s)))])
    res.update(xc_ms=lag_s * 1000.0, xc_peak=pk, xc_r0=r0, xc_lags=lags_s, xc_corr=corr)
    sh = shifted(t, cv, lag_s)
    ok = ~np.isnan(sh)
    res["r_before"] = float(np.corrcoef(imu[ok], cv[ok])[0, 1])
    res["r_after"] = float(np.corrcoef(imu[ok], sh[ok])[0, 1])
    res["cv_shifted"] = sh

    # ---- spike detection ----
    dist = max(1, int(round(a.min_dist / dt)))
    p_imu = find_peaks(imu, prominence=a.prom_frac * imu.max(), distance=dist)[0]
    p_cv = find_peaks(cv, prominence=a.prom_frac * cv.max(), distance=dist)[0]
    res.update(p_imu=p_imu, p_cv=p_cv)

    # ---- match every IMU spike with the CV spike expected `lag` later ----
    coarse = int(round(lag_s / dt))
    win = max(1, int(round(a.match_window / dt)))
    used, pairs, unmatched_imu = set(), [], []
    for k in p_imu:
        cand = np.array([c for c in p_cv if c not in used], dtype=int)
        if len(cand) == 0:
            unmatched_imu.append(int(k))
            continue
        j = int(cand[np.argmin(np.abs(cand - (k + coarse)))])
        if abs(j - (k + coarse)) > win:
            unmatched_imu.append(int(k))
            continue
        used.add(j)
        pairs.append((int(k), j))
    res.update(n_pairs=len(pairs), unmatched_imu=unmatched_imu,
               unmatched_cv=[int(c) for c in p_cv if c not in used], coarse_samples=coarse)
    if len(pairs) < 3:
        res.update(verdict="INCONCLUSIVE", reasons=[f"only {len(pairs)} spikes matched (need >= 3)"])
        return res

    base_i, base_c = float(np.median(imu)), float(np.median(cv))
    Ki = np.array([p[0] for p in pairs])
    Kc = np.array([p[1] for p in pairs])
    Ri = np.array([spike_centroid(imu, k, base_i) for k in Ki])
    Rc = np.array([spike_centroid(cv, k, base_c) for k in Kc])
    Ti, Tc = np.interp(Ri, idx_axis, t), np.interp(Rc, idx_axis, t)
    lag_ms = (Tc - Ti) * 1000.0
    lag_pk_ms = (Kc - Ki) * dt * 1000.0
    res.update(Ki=Ki, Kc=Kc, Ti=Ti, Tc=Tc, lag_ms=lag_ms, lag_pk_ms=lag_pk_ms,
               imu_amp=imu[Ki], cv_amp=cv[Kc])

    # ---- statistics with bootstrap confidence interval of the median ----
    rng = np.random.default_rng(0)
    boots = np.median(rng.choice(lag_ms, size=(5000, len(lag_ms))), axis=1)
    res["stats"] = dict(n=len(lag_ms), median_ms=float(np.median(lag_ms)), mean_ms=float(lag_ms.mean()),
                        std_ms=float(lag_ms.std(ddof=1)), min_ms=float(lag_ms.min()), max_ms=float(lag_ms.max()),
                        ci_low_ms=float(np.percentile(boots, 2.5)), ci_high_ms=float(np.percentile(boots, 97.5)),
                        median_peak_sample_ms=float(np.median(lag_pk_ms)))

    # ---- motion bursts (move / stop / move ...) ----
    burst = np.zeros(len(Ti), dtype=int)
    for e in range(1, len(Ti)):
        burst[e] = burst[e - 1] + (1 if Ti[e] - Ti[e - 1] > a.burst_gap else 0)
    n_b = int(burst.max()) + 1
    pad = int(round(1.0 / dt))
    bursts = []
    for b in range(n_b):
        m = burst == b
        i0 = max(0, int(min(Ki[m].min(), Kc[m].min())) - pad)
        i1 = min(len(t), int(max(Ki[m].max(), Kc[m].max())) + pad + 1)
        ls, pk_b, lgs, cr = xcorr_lag(imu[i0:i1], cv[i0:i1], dt, a.max_lag_burst)
        bursts.append(dict(burst=b + 1, i0=i0, i1=i1, t0=float(t[i0]), t1=float(t[i1 - 1]), n=int(m.sum()),
                           median_ms=float(np.median(lag_ms[m])), xc_ms=ls * 1000.0, xc_peak=pk_b,
                           lags_s=lgs, corr=cr))
    res.update(burst=burst, n_bursts=n_b, bursts=bursts)

    xcs = [b["xc_ms"] for b in bursts]
    res["burst_xc_range_ms"] = float(max(xcs) - min(xcs))
    return res


def conclude(res: dict, ang) -> None:
    """Combine the lag estimates into one best estimate, the verdict and a suggested fix."""
    dt_ms, tol = res["dt"] * 1000.0, res["tol_ms"]
    est = {"spike median": res["stats"]["median_ms"], "speed cross-correlation": res["xc_ms"]}
    if ang is not None:
        est["angle-vs-position cross-correlation"] = ang["best"]["lag_ms"]
    vals = list(est.values())
    best = float(np.median(vals))
    res.update(estimates=est, best_lag_ms=best, estimate_range_ms=float(max(vals) - min(vals)))
    reasons = ["lag estimates: " + "; ".join(f"{k} {v:+.0f} ms" for k, v in est.items())
               + f"  ->  best estimate (median of methods) {best:+.0f} ms"]
    if abs(best) <= tol:
        verdict = "SYNCHRONISED"
    else:
        verdict = "NOT SYNCHRONISED"
        who = "later than the IMU (CV is delayed)" if best > 0 else "earlier than the IMU (CV leads)"
        reasons.append(f"CV sees the motion {who} by about {abs(best):.0f} ms (tolerance +/-{tol:.0f} ms)")
        rng = (max(ang["burst_lag_ms"]) - min(ang["burst_lag_ms"])) if ang is not None else res["burst_xc_range_ms"]
        if rng <= tol:
            reasons.append(f"the delay is stable: per-burst lags differ by only {rng:.0f} ms, "
                           "so it behaves like a constant latency that can be compensated")
        else:
            reasons.append(f"the delay is NOT constant: per-burst lags differ by {rng:.0f} ms")
        shift = round(best / dt_ms)
        reasons.append(f"suggested fix: shift the CV data {abs(shift)} sample(s) ({abs(shift) * dt_ms:.0f} ms) "
                       f"{'earlier' if shift > 0 else 'later'} (or the IMU data the other way)")
    if res["estimate_range_ms"] > dt_ms:
        reasons.append(f"the methods differ by {res['estimate_range_ms']:.0f} ms (more than one sample), "
                       f"so treat the delay as a range of {min(vals):+.0f} to {max(vals):+.0f} ms")
    res.update(verdict=verdict, reasons=reasons)


def angle_check(t: np.ndarray, sig: dict, res: dict, a) -> dict:
    """Independent check: integrate the gyro into a head angle and correlate it with the
    CV position. The noisy CV data is not differentiated, so it does not share the
    weaknesses of the spike method. All axis pairs are tried; the best one is reported."""
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    hp = butter(2, a.hp_hz, "highpass", fs=fs, output="sos")
    lp = butter(2, min(a.lp_hz, 0.45 * fs), "lowpass", fs=fs, output="sos")

    def prep(x):  # zero-phase filters: remove drift and noise without adding any delay
        return sosfiltfilt(lp, sosfiltfilt(hp, x, axis=0), axis=0)

    # trapezoid rule: a plain cumsum()*dt would place the angle half a sample (25 ms at 20 Hz) too late
    ang = prep(cumulative_trapezoid(sig["imu_gyro"], dx=dt, initial=0, axis=0))
    pos = prep(sig["cv_fix"])
    rows = []
    for i in range(3):
        for j in range(3):
            best = None
            for sgn in (1.0, -1.0):          # a landmark axis may move opposite to the gyro axis
                lag_s, pk, lags_s, c = xcorr_lag(ang[:, i], sgn * pos[:, j], dt, a.max_lag)
                if lags_s is None:           # flat (zero-variance) signal: nothing to correlate
                    continue
                if best is None or pk > best[1]:
                    best = (lag_s, pk, lags_s, c, sgn)
            if best is None:
                continue
            lag_s, pk, lags_s, c, sgn = best
            rows.append(dict(gyro_axis="xyz"[i], cv_axis="xyz"[j], sign=sgn, i=i, j=j, lag_ms=lag_s * 1000.0,
                             peak=pk, r0=float(c[int(np.argmin(np.abs(lags_s)))]), lags_s=lags_s, corr=c))
    if not rows:
        raise SystemExit("angle check: no usable gyro/CV axis pair (all signals flat?)")
    rows.sort(key=lambda r: -r["peak"])
    b = rows[0]
    x, y = ang[:, b["i"]], b["sign"] * pos[:, b["j"]]
    sh = shifted(t, y, b["lag_ms"] / 1000.0)
    ok = ~np.isnan(sh)
    out = dict(best=b, top=rows[:4], x=x, y=y, y_shifted=sh,
               r_before=float(np.corrcoef(x[ok], y[ok])[0, 1]), r_after=float(np.corrcoef(x[ok], sh[ok])[0, 1]),
               burst_lag_ms=[])
    for bu in res.get("bursts", []):
        ls, _, _, _ = xcorr_lag(x[bu["i0"]:bu["i1"]], y[bu["i0"]:bu["i1"]], dt, a.max_lag_burst)
        out["burst_lag_ms"].append(ls * 1000.0)
    return out


# --------------------------------------------------------------------------
# log and tables
# --------------------------------------------------------------------------
def log_report(res: dict, a, ang=None) -> None:
    dt = res["dt"]
    log.info("")
    log.info("=" * 78)
    log.info("SPIKE DETECTION AND MATCHING")
    log.info("=" * 78)
    log.info(f"IMU spikes found: {len(res['p_imu'])}    CV spikes found: {len(res['p_cv'])}    "
             f"matched pairs: {res['n_pairs']}")
    log.info(f"coarse lag used for matching (from cross-correlation): {res['coarse_samples']} samples; "
             f"match window +/-{a.match_window:.2f} s")
    log.info(f"unmatched IMU spikes: {len(res['unmatched_imu'])}    unmatched CV spikes: {len(res['unmatched_cv'])}")
    if res["n_pairs"] < 3:
        log.info(f"\nVERDICT: {res['verdict']}  -  " + "; ".join(res["reasons"]))
        return
    log.info("")
    log.info("=" * 78)
    log.info("PER-SPIKE LAG  (ms, +ve = CV later than IMU; half-height centroid timing)")
    log.info("=" * 78)
    log.info(f"{'#':>3} {'burst':>5} {'t_imu(s)':>9} {'t_cv(s)':>9} {'lag(ms)':>9} {'peak-sample lag':>16}")
    for e in range(res["n_pairs"]):
        log.info(f"{e + 1:>3} {res['burst'][e] + 1:>5} {res['Ti'][e]:>9.3f} {res['Tc'][e]:>9.3f} "
                 f"{res['lag_ms'][e]:>+9.1f} {res['lag_pk_ms'][e]:>+16.0f}")
    s = res["stats"]
    log.info("")
    log.info("=" * 78)
    log.info(f"SUMMARY  (sample period = {dt * 1000:.1f} ms)")
    log.info("=" * 78)
    log.info(f"spike lag      : median {s['median_ms']:+.1f} ms  (95% bootstrap CI {s['ci_low_ms']:+.1f} to "
             f"{s['ci_high_ms']:+.1f})   mean {s['mean_ms']:+.1f}   std {s['std_ms']:.1f}   "
             f"range {s['min_ms']:+.0f} to {s['max_ms']:+.0f}")
    log.info(f"cross-corr lag : {res['xc_ms']:+.1f} ms  (whole record, sub-sample interpolated)")
    log.info(f"correlation    : {res['r_before']:.3f} with no shift  ->  {res['r_after']:.3f} after shifting CV "
             f"back by {res['xc_ms']:.0f} ms")
    log.info("")
    log.info("PER MOTION BURST")
    log.info(f"{'burst':>5} {'from(s)':>8} {'to(s)':>8} {'spikes':>7} {'spike median(ms)':>17} {'xcorr(ms)':>10}")
    for b in res["bursts"]:
        log.info(f"{b['burst']:>5} {b['t0']:>8.1f} {b['t1']:>8.1f} {b['n']:>7} {b['median_ms']:>+17.1f} {b['xc_ms']:>+10.1f}")
    log.info(f"per-burst cross-correlation lags differ by {res['burst_xc_range_ms']:.0f} ms")
    if ang is not None:
        b = ang["best"]
        log.info("")
        log.info("INDEPENDENT CHECK: integrated gyro angle vs CV position (CV data is not differentiated)")
        log.info(f"best pair      : gyro-{b['gyro_axis']} angle vs cv_{a.cv_point}_{b['cv_axis']} "
                 f"(sign {'+' if b['sign'] > 0 else '-'})  ->  CV lag {b['lag_ms']:+.1f} ms")
        log.info(f"correlation    : {ang['r_before']:.3f} with no shift  ->  {ang['r_after']:.3f} at the best lag")
        log.info("other pairs    : " + "; ".join(f"gyro-{r['gyro_axis']}/cv_{r['cv_axis']} {r['lag_ms']:+.0f} ms "
                                                  f"(r={r['peak']:.2f})" for r in ang["top"][1:]))
        log.info("per burst      : " + ", ".join(f"burst {k + 1}: {v:+.0f} ms" for k, v in enumerate(ang["burst_lag_ms"])))
    log.info("")
    log.info("=" * 78)
    log.info(f"VERDICT: {res['verdict']}   (tolerance +/-{res['tol_ms']:.0f} ms)")
    for r in res["reasons"]:
        log.info(f"  - {r}")
    log.info("=" * 78)


def write_tables(res: dict, t, sig, outdir: Path) -> None:
    if res["n_pairs"] < 3:
        return
    pd.DataFrame({"spike": np.arange(1, res["n_pairs"] + 1), "burst": res["burst"] + 1,
                  "t_imu_s": res["Ti"].round(4), "t_cv_s": res["Tc"].round(4),
                  "lag_ms": res["lag_ms"].round(2), "peak_sample_lag_ms": res["lag_pk_ms"].round(1),
                  "imu_peak_value": res["imu_amp"].round(4), "cv_peak_value": res["cv_amp"].round(4)}
                 ).to_csv(outdir / "spike_lags.csv", index=False)
    pd.DataFrame([{"burst": b["burst"], "from_s": round(b["t0"], 2), "to_s": round(b["t1"], 2),
                   "spikes": b["n"], "spike_median_lag_ms": round(b["median_ms"], 2),
                   "xcorr_lag_ms": round(b["xc_ms"], 2), "xcorr_peak": round(b["xc_peak"], 4)}
                  for b in res["bursts"]]).to_csv(outdir / "burst_summary.csv", index=False)
    pd.DataFrame({"lag_ms": (res["xc_lags"] * 1000).round(1), "correlation": res["xc_corr"].round(5)}
                 ).to_csv(outdir / "xcorr_curve.csv", index=False)
    pd.DataFrame({"t_s": t.round(3), "imu_activity": sig["imu"], "cv_speed": sig["cv"],
                  "cv_speed_shifted_back": res["cv_shifted"]}).to_csv(outdir / "signals.csv", index=False)


def write_json(res: dict, outdir: Path, meta: dict, ang=None) -> None:
    out = {"meta": meta, "verdict": res["verdict"], "reasons": res["reasons"], "tolerance_ms": res["tol_ms"],
           "sample_period_ms": res["dt"] * 1000.0, "spikes_matched": res["n_pairs"],
           "xcorr_lag_ms": res["xc_ms"], "corr_no_shift": res["r_before"], "corr_after_shift": res["r_after"]}
    if res["n_pairs"] >= 3:
        out["spike_lag"] = res["stats"]
        out["bursts"] = [{k: v for k, v in b.items() if k not in ("lags_s", "corr")} for b in res["bursts"]]
        out["burst_xcorr_range_ms"] = res["burst_xc_range_ms"]
        out["lag_estimates_ms"] = res["estimates"]
        out["best_lag_ms"] = res["best_lag_ms"]
    if ang is not None:
        b = ang["best"]
        out["angle_check"] = {"gyro_axis": b["gyro_axis"], "cv_axis": b["cv_axis"], "sign": b["sign"],
                              "lag_ms": b["lag_ms"], "corr_no_shift": ang["r_before"],
                              "corr_at_best_lag": ang["r_after"], "burst_lag_ms": ang["burst_lag_ms"]}
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def norm(y: np.ndarray) -> np.ndarray:
    return y / np.nanmax(y)


def shade_bursts(ax, res, t):
    for b in res["bursts"]:
        ax.axvspan(b["t0"], b["t1"], color="0.5", alpha=0.10, lw=0)


def plot_overview(t, sig, res, a, path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
    ax = axes[0]
    for j, x in enumerate("xyz"):
        ax.plot(t, sig["cv_raw"][:, j] - np.nanmean(sig["cv_raw"][:, j]), lw=1.2, color=CV_AX[j], label=f"cv_{a.cv_point}_{x}")
    ax.set_title(f"CV landmark {a.cv_point}: position (mean removed)", loc="left", fontsize=11)
    ax = axes[1]
    for j, x in enumerate("xyz"):
        ax.plot(t, sig["imu_gyro"][:, j], lw=1.2, color=IMU_AX[j], label=f"IMU {a.imu} g{x}")
    ax.set_title(f"IMU {a.imu}: raw gyro", loc="left", fontsize=11)
    ax = axes[2]
    ax.plot(t, norm(sig["imu"]), color=IMU_C, lw=1.5, label=f"IMU {a.imu} activity")
    ax.plot(t, norm(sig["cv"]), color=CV_C, lw=1.5, label=f"CV {a.cv_point} speed")
    if res["n_pairs"] >= 3:
        ax.plot(t[res["p_imu"]], norm(sig["imu"])[res["p_imu"]] + 0.05, "v", color=IMU_C, ms=5, label="IMU spikes")
        ax.plot(t[res["p_cv"]], norm(sig["cv"])[res["p_cv"]] + 0.05, "v", color=CV_C, ms=5, mfc="none", label="CV spikes")
    ax.set_title("Motion activity, each scaled to its own maximum (used for spike detection)", loc="left", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.25)
        if res["n_pairs"] >= 3:
            shade_bursts(ax, res, t)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, frameon=False, fontsize=9)
    if res["n_pairs"] >= 3:
        for b in res["bursts"]:
            axes[0].text((b["t0"] + b["t1"]) / 2, 0.97, f"{b['burst']}", ha="center", va="top", fontsize=9,
                         color="0.35", transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("CV vs IMU: do they see the head motion at the same time?  (grey bands / numbers = motion bursts)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zoom(t, sig, res, a, path):
    n = res["n_bursts"]
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), squeeze=False)
    imu_n, cv_n = norm(sig["imu"]), norm(sig["cv"])
    for c in range(nrows * ncols):
        ax = axs[c // ncols][c % ncols]
        if c >= n:
            ax.axis("off")
            continue
        b = res["bursts"][c]
        sl = slice(b["i0"], b["i1"])
        ax.plot(t[sl], imu_n[sl], "-o", ms=2.6, lw=1.1, color=IMU_C, label=f"IMU {a.imu}")
        ax.plot(t[sl], cv_n[sl], "-o", ms=2.6, lw=1.1, color=CV_C, label=f"CV {a.cv_point}")
        m = res["burst"] == c
        for ti in res["Ti"][m]:
            ax.axvline(ti, color=IMU_C, ls="--", lw=0.8, alpha=0.7)
        for tc in res["Tc"][m]:
            ax.axvline(tc, color=CV_C, ls=":", lw=1.0, alpha=0.9)
        ax.set_title(f"Burst {b['burst']}  ({b['t0']:.1f}-{b['t1']:.1f} s):  spike median {b['median_ms']:+.0f} ms, "
                     f"xcorr {b['xc_ms']:+.0f} ms", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.25)
        if c == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Zoom on every motion burst (dots = samples; dashed blue = IMU spike, dotted orange = CV spike)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_angle(t, ang, res, a, path):
    b = ang["best"]
    sd = float(np.nanstd(ang["y"]))
    x, y, ys = ang["x"] / float(np.nanstd(ang["x"])), ang["y"] / sd, ang["y_shifted"] / sd
    lab_i = f"IMU {a.imu} gyro-{b['gyro_axis']} integrated (head angle)"
    lab_c = f"CV cv_{a.cv_point}_{b['cv_axis']} ({'+' if b['sign'] > 0 else '-'})"
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ax, yy, lab, title in [(axes[0], y, f"{lab_c} as logged", f"As logged (correlation {ang['r_before']:.2f})"),
                               (axes[1], ys, f"{lab_c} shifted back {b['lag_ms']:.0f} ms",
                                f"After shifting CV back by {b['lag_ms']:.0f} ms (correlation {ang['r_after']:.2f})")]:
        ax.plot(t, x, color=IMU_C, lw=1.7, label=lab_i)
        ax.plot(t, yy, color=CV_C, lw=1.4, label=lab)
        shade_bursts(ax, res, t)
        ax.set_xlim(t[0], t[-1])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("scaled to unit std")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)
    axes[1].set_xlabel("time (s)")
    ax = axes[2]
    for k, r in enumerate(ang["top"]):
        m = np.abs(r["lags_s"]) <= 0.8
        ax.plot(r["lags_s"][m] * 1000, r["corr"][m], "-o" if k == 0 else "-", ms=3, lw=2.2 if k == 0 else 1.0,
                color="k" if k == 0 else BURST_C[k], alpha=1.0 if k == 0 else 0.8,
                label=f"gyro-{r['gyro_axis']} / cv_{a.cv_point}_{r['cv_axis']}: {r['lag_ms']:+.0f} ms (r = {r['peak']:.2f})")
    ax.axvline(b["lag_ms"], color="k", ls="--", lw=1)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("lag of CV vs IMU (ms)")
    ax.set_ylabel("normalised cross-correlation")
    ax.set_title("Cross-correlation of angle vs position (best pairs)", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.suptitle("Independent check: integrated gyro angle vs CV position (grey = motion bursts)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lags(t, sig, res, a, path):
    s, tol = res["stats"], res["tol_ms"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    ax.axhspan(-tol, tol, color="0.5", alpha=0.15, lw=0, label=f"+/-1 sample ({tol:.0f} ms)")
    for b in range(res["n_bursts"]):
        m = res["burst"] == b
        ax.plot(res["Ti"][m], res["lag_ms"][m], "o", ms=5, color=BURST_C[b % len(BURST_C)], label=f"burst {b + 1}")
    ax.axhspan(s["ci_low_ms"], s["ci_high_ms"], color=CV_C, alpha=0.18, lw=0)
    ax.axhline(s["median_ms"], color=CV_C, lw=1.6, label=f"median {s['median_ms']:+.0f} ms")
    ax.axhline(res["xc_ms"], color="k", ls="--", lw=1.2, label=f"cross-corr {res['xc_ms']:+.0f} ms")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("time of IMU spike (s)")
    ax.set_ylabel("lag of CV vs IMU (ms)")
    ax.set_title("Lag of every spike (orange band = 95% CI of median)", loc="left")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[0, 1]
    lo, hi = math.floor(s["min_ms"] / 25) * 25, math.ceil(s["max_ms"] / 25) * 25 + 25
    ax.hist(res["lag_ms"], bins=np.arange(lo, hi + 1, 25), color=CV_C, alpha=0.75, edgecolor="white")
    ax.axvline(s["median_ms"], color=CV_C, lw=2, label=f"median {s['median_ms']:+.0f} ms")
    ax.axvline(res["xc_ms"], color="k", ls="--", lw=1.4, label=f"cross-corr {res['xc_ms']:+.0f} ms")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("lag of CV vs IMU (ms)")
    ax.set_ylabel("number of spikes")
    ax.set_title("Distribution of spike lags", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    ax = axs[1, 0]
    ml = np.abs(res["xc_lags"]) <= 0.8
    ax.plot(res["xc_lags"][ml] * 1000, res["xc_corr"][ml], "-o", ms=3, lw=2.2, color="k", label="whole record")
    for b in res["bursts"]:
        mb = np.abs(b["lags_s"]) <= 0.8
        ax.plot(b["lags_s"][mb] * 1000, b["corr"][mb], lw=1.0, color=BURST_C[(b["burst"] - 1) % len(BURST_C)],
                alpha=0.9, label=f"burst {b['burst']}: {b['xc_ms']:+.0f} ms")
    ax.axvline(res["xc_ms"], color="k", ls="--", lw=1)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("lag of CV vs IMU (ms)")
    ax.set_ylabel("normalised cross-correlation")
    ax.set_title(f"Cross-correlation (whole record peak at {res['xc_ms']:+.0f} ms)", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[1, 1]
    b0 = res["bursts"][0]
    sl = slice(b0["i0"], b0["i1"])
    ax.plot(t[sl], norm(sig["imu"])[sl], color=IMU_C, lw=1.8, label=f"IMU {a.imu}")
    ax.plot(t[sl], norm(sig["cv"])[sl], color=CV_C, lw=1.2, ls="--", alpha=0.7,
            label=f"CV {a.cv_point} as logged (r = {res['r_before']:.2f})")
    ax.plot(t[sl], (res["cv_shifted"] / np.nanmax(sig["cv"]))[sl], color=CV_C, lw=1.6,
            label=f"CV shifted back {res['xc_ms']:.0f} ms (r = {res['r_after']:.2f})")
    ax.set_xlabel("time (s)")
    ax.set_title("Before / after shifting CV (burst 1; r is for the whole record)", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle(f"CV vs IMU lag analysis  -  verdict: {res['verdict']}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# self-test: synthetic head swings with a known CV delay
# --------------------------------------------------------------------------
def selftest(a) -> int:
    fs, dur = 20.0, 45.0
    t = np.arange(0, dur, 1 / fs)
    tf = np.arange(0, dur, 0.001)
    env = np.zeros_like(tf)
    for s0, s1 in [(4.0, 11.0), (17.0, 30.0), (38.0, 43.0)]:
        ramp = np.clip(np.minimum(tf - s0, s1 - tf) / 0.5, 0, 1)
        env = np.maximum(env, ramp)
    theta = 0.5 * env * np.sin(2 * np.pi * 1.0 * tf)           # head angle (rad)
    omega = np.gradient(theta, 0.001)                           # angular rate (rad/s)
    rng = np.random.default_rng(3)
    ok_all = True
    print("selftest: true CV delay vs recovered (ms)")
    for d in (0.160, 0.0, -0.060):
        th_cv = np.interp(t - d, tf, theta)
        cvx = np.round(-0.01 + 0.005 * th_cv, 4)
        cvy = np.round(0.6 + 0.3 * np.sin(th_cv) + rng.normal(0, 0.0003, len(t)), 4)
        cvz = np.round(-0.2 + 0.3 * (1 - np.cos(th_cv)) + rng.normal(0, 0.0003, len(t)), 4)
        cvm = np.c_[cvx, cvy, cvz]
        stale_rows = rng.choice(np.arange(1, len(t)), size=int(0.04 * len(t)), replace=False)
        for r in stale_rows:                                    # 4 % repeated camera frames
            cvm[r] = cvm[r - 1]
        ip = f"{a.imu_prefix}{a.imu}_"
        df = pd.DataFrame({a.time_col: 1.79e9 + t, f"cv_{a.cv_point}_x": cvm[:, 0],
                           f"cv_{a.cv_point}_y": cvm[:, 1], f"cv_{a.cv_point}_z": cvm[:, 2],
                           ip + "ax": -1.0, ip + "ay": 0.0, ip + "az": 0.0, ip + "gx": 0.0, ip + "gy": 0.0,
                           ip + "gz": np.interp(t, tf, omega) + rng.normal(0, 0.02, len(t))})
        sig = build_signals(df, t, a)
        res = analyse(t, sig["imu"], sig["cv"], a)
        if res["n_pairs"] < 3:
            print(f"  true {d * 1000:+6.0f}: FAIL (only {res['n_pairs']} spikes matched)")
            ok_all = False
            continue
        ang = angle_check(t, sig, res, a)
        good = (abs(res["stats"]["median_ms"] - d * 1000) < 25 and abs(res["xc_ms"] - d * 1000) < 25
                and abs(ang["best"]["lag_ms"] - d * 1000) < 25)
        ok_all &= good
        print(f"  true {d * 1000:+6.0f}  spikes {res['stats']['median_ms']:+7.1f}  xcorr {res['xc_ms']:+7.1f}  "
              f"angle-check {ang['best']['lag_ms']:+7.1f}  matched {res['n_pairs']}  {'OK' if good else 'FAIL'}")
    print("selftest:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Check synchronisation of a CV landmark and an IMU from spike lags.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("csv", nargs="?", help="input CSV file")
    p.add_argument("--outdir", default="cv_imu_results", help="output folder")
    p.add_argument("--cv-point", type=int, default=0, help="CV landmark number (columns cv_<n>_x,y,z)")
    p.add_argument("--imu", type=int, default=1, help="IMU number")
    p.add_argument("--imu-prefix", default="ESP32_IMU_", help="IMU column prefix; columns are <prefix><n>_ax..gz")
    p.add_argument("--time-col", default="sys_time", help="time column (seconds)")
    p.add_argument("--hp-hz", type=float, default=0.3, help="angle check: high-pass cut-off removing drift (Hz)")
    p.add_argument("--lp-hz", type=float, default=4.0, help="angle check: low-pass cut-off removing noise (Hz)")
    p.add_argument("--smooth", type=int, default=3, help="moving-average length in samples (both signals; 1 = off)")
    p.add_argument("--no-stale-fix", action="store_true", help="do not interpolate repeated CV rows")
    p.add_argument("--prom-frac", type=float, default=0.25, help="spike prominence = frac x signal maximum")
    p.add_argument("--min-dist", type=float, default=0.2, help="minimum time between spikes (s)")
    p.add_argument("--match-window", type=float, default=0.25, help="match window around the expected CV spike (s)")
    p.add_argument("--burst-gap", type=float, default=2.0, help="gap (s) that separates motion bursts")
    p.add_argument("--max-lag", type=float, default=1.0, help="max lag searched, whole record (s)")
    p.add_argument("--max-lag-burst", type=float, default=0.5, help="max lag searched, per burst (s)")
    p.add_argument("--tol-ms", type=float, default=0.0, help="sync tolerance in ms (0 = one sample period)")
    p.add_argument("--selftest", action="store_true", help="run a synthetic-data self-test and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.selftest:
        return selftest(a)
    if not a.csv:
        raise SystemExit("Give a CSV file, or use --selftest")
    csv_path = Path(a.csv)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)

    df, t = load_data(csv_path, a)
    dts = np.diff(t)
    dt = float(np.median(dts))
    sig = build_signals(df, t, a)

    log.info("=" * 78)
    log.info("CV vs IMU SYNCHRONISATION CHECK")
    log.info("=" * 78)
    log.info(f"file         : {csv_path.name}")
    log.info(f"rows         : {len(df)}   duration: {t[-1]:.2f} s")
    if "local_time" in df.columns:
        log.info(f"local time   : {df['local_time'].iloc[0]}  ->  {df['local_time'].iloc[-1]}")
    log.info(f"sample step  : median {dt * 1000:.2f} ms ({1 / dt:.2f} Hz), min {dts.min() * 1000:.2f}, "
             f"max {dts.max() * 1000:.2f}, std {dts.std() * 1000:.3f} ms; gaps > 1.5x step: {int((dts > 1.5 * dt).sum())}")
    log.info(f"CV landmark  : cv_{a.cv_point}   IMU: {a.imu}   signals: CV speed vs gyro magnitude   "
             f"smoothing: {a.smooth} samples")
    n_stale = int(sig["stale"].sum())
    log.info(f"repeated CV rows (stale camera frames): {n_stale} of {len(df)} "
             f"({100 * n_stale / len(df):.1f}%) -> {'left as is' if a.no_stale_fix else 'interpolated'}")
    log.info(f"lag resolution: one sample = {dt * 1000:.1f} ms (sub-sample values are interpolated estimates)")

    res = analyse(t, sig["imu"], sig["cv"], a)
    ang = angle_check(t, sig, res, a) if res["n_pairs"] >= 3 else None
    if res["n_pairs"] >= 3:
        conclude(res, ang)
    log_report(res, a, ang)
    write_tables(res, t, sig, outdir)
    write_json(res, outdir, {"file": csv_path.name, "rows": len(df), "duration_s": float(t[-1]),
                             "cv_point": a.cv_point, "imu": a.imu, "stale_cv_rows": n_stale}, ang)
    if res["n_pairs"] >= 3:
        plot_overview(t, sig, res, a, outdir / "fig1_overview.png")
        plot_zoom(t, sig, res, a, outdir / "fig2_zoom_bursts.png")
        plot_lags(t, sig, res, a, outdir / "fig3_lag_analysis.png")
        plot_angle(t, ang, res, a, outdir / "fig4_angle_check.png")
    log.info(f"\nResults written to: {outdir.resolve()}")
    return 0 if res["verdict"] == "SYNCHRONISED" else 2


if __name__ == "__main__":
    sys.exit(main())
