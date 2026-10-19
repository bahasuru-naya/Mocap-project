#!/usr/bin/env python3
"""
imu_sync_check.py
=================
Check whether several IMUs logged in one CSV file are time-synchronised.

Test idea: all IMUs are bound together and moved as one body (move -> stop ->
move -> stop ...). Every sharp movement produces a "spike" that all IMUs must
see at the same moment. This script

  1. finds the spikes in every IMU,
  2. matches the same physical spike across IMUs,
  3. measures the lag (delay) of each IMU relative to a reference IMU,
  4. cross-checks the lag with cross-correlation,
  5. writes figures, log files and CSV/JSON results, and gives a verdict.

Sign convention:  lag > 0  =>  that IMU sees the spike LATER than the reference.

Usage
-----
    python3 imu_sync_check.py data.csv
    python3 imu_sync_check.py data.csv --ref 3 --signal accel --outdir out
    python3 imu_sync_check.py --selftest

See README.md for details.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import correlate, find_peaks  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#000000", "#F0E442"]
log = logging.getLogger("imu_sync")


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
    """Robust sub-sample spike time (in samples).

    Centroid of the part of the spike that is above half height (measured from
    the resting level `base`), in the contiguous region around the peak sample k.
    Unlike the raw peak sample, it does not jump when a spike has a flat top.
    """
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
    Returns (lag_s, peak_corr, lags_s, corr).
    """
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


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_data(csv_path: Path, prefix: str, time_col: str):
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise SystemExit(f"Time column '{time_col}' not found in {csv_path}")
    pat = re.compile(rf"{re.escape(prefix)}(\d+)_gx$")
    ids = sorted({int(m.group(1)) for c in df.columns if (m := pat.match(c))})
    if len(ids) < 2:
        raise SystemExit(f"Need at least 2 IMUs with columns like '{prefix}<n>_gx'; found {ids}")
    t_abs = df[time_col].to_numpy(float)
    return df, t_abs - t_abs[0], ids


def build_signals(df: pd.DataFrame, ids, prefix: str, kind: str):
    """One scalar 'activity' signal per IMU, independent of mounting orientation."""
    sig = {}
    for i in ids:
        if kind == "gyro":
            v = df[[f"{prefix}{i}_g{a}" for a in "xyz"]].to_numpy(float)
            sig[i] = np.linalg.norm(v, axis=1)
        else:  # accel: remove gravity/offset by subtracting the median magnitude
            v = df[[f"{prefix}{i}_a{a}" for a in "xyz"]].to_numpy(float)
            mag = np.linalg.norm(v, axis=1)
            sig[i] = np.abs(mag - np.median(mag))
    return sig


# --------------------------------------------------------------------------
# core analysis
# --------------------------------------------------------------------------
def analyse(t: np.ndarray, sig: dict, a) -> dict:
    ids = list(sig.keys())
    dt = float(np.median(np.diff(t)))
    ref = a.ref if a.ref in ids else ids[0]
    rc = ids.index(ref)
    tol_ms = a.tol_ms if a.tol_ms else dt * 1000.0

    prom = a.prominence or a.prom_frac * float(np.median([y.max() for y in sig.values()]))
    dist = max(1, int(round(a.min_dist / dt)))
    peaks = {i: find_peaks(y, prominence=prom, distance=dist)[0] for i, y in sig.items()}

    # ---- match spikes of every IMU to the reference IMU's spikes ----
    win = max(1, int(round(a.match_window / dt)))
    used = {i: set() for i in ids}
    events, unmatched = [], []
    for kr in peaks[ref]:
        row, ok = {ref: int(kr)}, True
        for i in ids:
            if i == ref:
                continue
            cand = np.array([c for c in peaks[i] if c not in used[i]], dtype=int)
            if len(cand) == 0:
                ok = False
                break
            j = int(cand[np.argmin(np.abs(cand - kr))])
            if abs(j - kr) > win:
                ok = False
                break
            row[i] = j
        if ok:
            for i, j in row.items():
                used[i].add(j)
            events.append(row)
        else:
            unmatched.append(int(kr))
    n_ev = len(events)
    extra = {i: int(len(peaks[i]) - len(used[i])) for i in ids if i != ref}

    res = dict(ids=ids, ref=ref, rc=rc, dt=dt, fs=1.0 / dt, tol_ms=tol_ms, prominence=prom,
               n_peaks={i: int(len(peaks[i])) for i in ids}, n_events=n_ev,
               unmatched_ref_peaks=unmatched, extra_peaks=extra, peaks=peaks)
    if n_ev == 0:
        res["verdict"], res["reasons"] = "INCONCLUSIVE", ["no spikes matched across all IMUs"]
        return res

    idx_axis = np.arange(len(t))
    base = {i: float(np.median(sig[i])) for i in ids}                        # resting level
    K = np.array([[e[i] for i in ids] for e in events], dtype=int)           # integer peak samples
    R = np.array([[spike_centroid(sig[i], e[i], base[i]) for i in ids] for e in events])  # refined samples
    T = np.interp(R, idx_axis, t)                                           # refined peak times (s)
    lag_int_ms = (K - K[:, [rc]]) * dt * 1000.0
    lag_ms = (R - R[:, [rc]]) * dt * 1000.0
    spread_ms = (R.max(axis=1) - R.min(axis=1)) * dt * 1000.0
    spread_int_ms = (K.max(axis=1) - K.min(axis=1)) * dt * 1000.0

    # ---- group spikes into motion cycles (move/stop repetitions) ----
    tref = T[:, rc]
    cycle = np.zeros(n_ev, dtype=int)
    for e in range(1, n_ev):
        cycle[e] = cycle[e - 1] + (1 if tref[e] - tref[e - 1] > a.cycle_gap else 0)
    n_cyc = int(cycle.max()) + 1
    pad = int(round(0.5 / dt))
    windows = []
    for c in range(n_cyc):
        ks = K[cycle == c]
        windows.append((max(0, int(ks.min()) - pad), min(len(t), int(ks.max()) + pad + 1)))

    # ---- cross-correlation (whole record and per cycle) ----
    xc_whole, xc_curves, xc_cycle = {}, {}, {}
    for i in ids:
        if i == ref:
            continue
        lag_s, pk, lags_s, c = xcorr_lag(sig[ref], sig[i], dt, a.max_lag)
        xc_whole[i] = (lag_s * 1000.0, pk)
        xc_curves[i] = (lags_s, c)
        xc_cycle[i] = []
        for (i0, i1) in windows:
            ls, pk_c, _, _ = xcorr_lag(sig[ref][i0:i1], sig[i][i0:i1], dt, a.max_lag_cycle)
            xc_cycle[i].append((ls * 1000.0, pk_c))

    # ---- per-IMU statistics ----
    stats = {}
    for c_i, i in enumerate(ids):
        lm, li = lag_ms[:, c_i], lag_int_ms[:, c_i]
        d = dict(n=n_ev, mean_ms=float(lm.mean()), median_ms=float(np.median(lm)),
                 std_ms=float(lm.std(ddof=1)) if n_ev > 1 else 0.0,
                 min_ms=float(lm.min()), max_ms=float(lm.max()),
                 pct_peak_same_sample=float(100.0 * np.mean(li == 0)),
                 pct_within_tol=float(100.0 * np.mean(np.abs(lm) <= tol_ms + 1e-9)),
                 max_abs_peak_sample_ms=float(np.abs(li).max()))
        if n_ev >= 3:
            d["drift_ms_per_min"] = float(theilslopes(lm, tref / 60.0)[0])   # robust slope
        else:
            d["drift_ms_per_min"] = float("nan")
        if i != ref:
            d["xcorr_ms"], d["xcorr_peak"] = xc_whole[i]
        stats[i] = d

    # ---- pairwise median lag matrix: entry [a, b] = how much later b sees spikes than a ----
    pair = np.array([[float(np.median((R[:, b] - R[:, aa]) * dt * 1000.0)) for b in range(len(ids))]
                     for aa in range(len(ids))])

    # ---- verdict ----
    reasons = []
    if n_ev < 3:
        verdict = "INCONCLUSIVE"
        reasons.append(f"only {n_ev} spikes matched across all IMUs (need >= 3)")
    else:
        bad = False
        for i in ids:
            if i == ref:
                continue
            s = stats[i]
            if abs(s["median_ms"]) > tol_ms:
                bad = True
                reasons.append(f"IMU {i}: median lag {s['median_ms']:+.1f} ms exceeds +/-{tol_ms:.0f} ms")
            if s["pct_within_tol"] < 90.0:
                bad = True
                reasons.append(f"IMU {i}: only {s['pct_within_tol']:.0f}% of spikes within +/-{tol_ms:.0f} ms")
            if abs(s["xcorr_ms"]) > tol_ms:
                bad = True
                reasons.append(f"IMU {i}: cross-correlation lag {s['xcorr_ms']:+.1f} ms exceeds +/-{tol_ms:.0f} ms")
        if unmatched:
            reasons.append(f"{len(unmatched)} reference spike(s) had no match in every IMU (check --match-window)")
        verdict = "NOT SYNCHRONISED" if bad else "SYNCHRONISED"
    res.update(K=K, R=R, T=T, lag_ms=lag_ms, lag_int_ms=lag_int_ms, spread_ms=spread_ms,
               spread_int_ms=spread_int_ms, cycle=cycle, n_cycles=n_cyc, windows=windows,
               xc_whole=xc_whole, xc_curves=xc_curves, xc_cycle=xc_cycle, stats=stats,
               pair=pair, verdict=verdict, reasons=reasons)
    return res


# --------------------------------------------------------------------------
# outputs: log, tables
# --------------------------------------------------------------------------
def log_report(res: dict) -> None:
    ids, ref, dt = res["ids"], res["ref"], res["dt"]
    log.info("")
    log.info("=" * 78)
    log.info("SPIKE DETECTION")
    log.info("=" * 78)
    log.info(f"prominence threshold : {res['prominence']:.3f}  (signal units)")
    log.info("spikes found per IMU: " + ", ".join(f"IMU{i}={res['n_peaks'][i]}" for i in ids))
    log.info(f"reference IMU        : {ref}")
    log.info(f"spikes matched in ALL IMUs: {res['n_events']}"
             f"   (unmatched reference spikes: {len(res['unmatched_ref_peaks'])})")
    if any(res["extra_peaks"].values()):
        log.info("extra unmatched spikes: " + ", ".join(f"IMU{i}={n}" for i, n in res["extra_peaks"].items()))
    if res["n_events"] == 0:
        log.info(f"\nVERDICT: {res['verdict']}  -  " + "; ".join(res["reasons"]))
        return
    T, lag_ms, spread = res["T"], res["lag_ms"], res["spread_ms"]
    log.info("")
    log.info("=" * 78)
    log.info(f"PER-SPIKE LAG vs IMU {ref}  (ms, +ve = later than reference; half-height centroid timing)")
    log.info("=" * 78)
    head = f"{'#':>3} {'cyc':>3} {'t_ref(s)':>9} " + " ".join(f"{'IMU'+str(i):>8}" for i in ids) + f" {'spread':>8}"
    log.info(head)
    for e in range(res["n_events"]):
        row = f"{e + 1:>3} {res['cycle'][e] + 1:>3} {T[e, res['rc']]:>9.3f} "
        row += " ".join(f"{lag_ms[e, c]:>+8.1f}" for c in range(len(ids)))
        row += f" {spread[e]:>8.1f}"
        log.info(row)
    log.info("")
    log.info("=" * 78)
    log.info(f"PER-IMU SUMMARY vs IMU {ref}  (sample period = {dt * 1000:.1f} ms)")
    log.info("=" * 78)
    log.info(f"{'IMU':>4} {'median':>8} {'mean':>8} {'std':>7} {'min':>8} {'max':>8} "
             f"{'pk=same%':>8} {'in tol%':>8} {'xcorr':>8} {'drift/min':>10}")
    for i in ids:
        s = res["stats"][i]
        xc = f"{s['xcorr_ms']:+8.1f}" if "xcorr_ms" in s else f"{'ref':>8}"
        log.info(f"{i:>4} {s['median_ms']:>+8.1f} {s['mean_ms']:>+8.1f} {s['std_ms']:>7.1f} "
                 f"{s['min_ms']:>+8.1f} {s['max_ms']:>+8.1f} {s['pct_peak_same_sample']:>8.0f} "
                 f"{s['pct_within_tol']:>8.0f} {xc} {s['drift_ms_per_min']:>+10.1f}")
    log.info("(lag values in ms. pk=same% = share of spikes whose raw peak falls on the same sample as the")
    log.info(" reference; in tol% = share within tolerance by centroid timing; drift/min = robust (Theil-Sen)")
    log.info(" slope of lag vs time in ms per minute - indicative only on a short recording)")
    log.info("")
    log.info("Spike-time spread across all IMUs (latest minus earliest): "
             f"median {np.median(spread):.1f} ms, max {spread.max():.1f} ms")
    log.info(f"Raw peak-sample spread (quantised, flat-topped spikes can hop 1-2 samples): "
             f"max {res['spread_int_ms'].max():.0f} ms")
    log.info("")
    log.info("PAIRWISE MEDIAN LAG (ms): row IMU -> column IMU  (+ve = column sees spikes later)")
    log.info("      " + " ".join(f"{'IMU' + str(i):>8}" for i in ids))
    for r_i, i in enumerate(ids):
        log.info(f"IMU{i:<3}" + " ".join(f"{res['pair'][r_i, c]:>+8.1f}" for c in range(len(ids))))
    log.info("")
    log.info("CROSS-CORRELATION LAG PER MOTION CYCLE (ms vs reference)")
    log.info(f"{'cycle':>6} " + " ".join(f"{'IMU' + str(i):>8}" for i in ids if i != ref))
    for c in range(res["n_cycles"]):
        log.info(f"{c + 1:>6} " + " ".join(f"{res['xc_cycle'][i][c][0]:>+8.1f}" for i in ids if i != ref))
    log.info("")
    log.info("=" * 78)
    log.info(f"VERDICT: {res['verdict']}   (tolerance +/-{res['tol_ms']:.0f} ms)")
    for r in res["reasons"]:
        log.info(f"  - {r}")
    log.info("=" * 78)


def write_tables(res: dict, outdir: Path) -> None:
    if res["n_events"] == 0:
        return
    ids, rc, dt = res["ids"], res["rc"], res["dt"]
    rows = []
    for e in range(res["n_events"]):
        r = {"spike": e + 1, "cycle": int(res["cycle"][e]) + 1,
             "t_ref_s": round(float(res["T"][e, rc]), 4),
             "spread_ms": round(float(res["spread_ms"][e]), 2)}
        for c, i in enumerate(ids):
            r[f"imu{i}_peak_sample"] = int(res["K"][e, c])
            r[f"imu{i}_peak_time_s"] = round(float(res["T"][e, c]), 4)
            r[f"imu{i}_lag_samples"] = int(round(res["lag_int_ms"][e, c] / (dt * 1000.0)))
            r[f"imu{i}_lag_ms"] = round(float(res["lag_ms"][e, c]), 2)
        rows.append(r)
    pd.DataFrame(rows).to_csv(outdir / "spike_lags.csv", index=False)

    srows = []
    for i in ids:
        s = res["stats"][i]
        srows.append({"imu": i, "is_reference": i == res["ref"], **{k: round(v, 3) for k, v in s.items()}})
    pd.DataFrame(srows).to_csv(outdir / "lag_summary.csv", index=False)

    xrows = []
    for i in ids:
        if i == res["ref"]:
            continue
        xrows.append({"scope": "whole record", "imu": i,
                      "lag_ms": round(res["xc_whole"][i][0], 2), "peak_corr": round(res["xc_whole"][i][1], 4)})
        for c, (ls, pk) in enumerate(res["xc_cycle"][i]):
            xrows.append({"scope": f"cycle {c + 1}", "imu": i, "lag_ms": round(ls, 2), "peak_corr": round(pk, 4)})
    pd.DataFrame(xrows).to_csv(outdir / "xcorr_lags.csv", index=False)

    lab = [f"IMU{i}" for i in ids]
    pd.DataFrame(np.round(res["pair"], 2), index=lab, columns=lab).to_csv(outdir / "pairwise_lag_ms.csv")


def write_json(res: dict, outdir: Path, meta: dict) -> None:
    out = {"meta": meta, "verdict": res["verdict"], "reasons": res["reasons"],
           "reference_imu": res["ref"], "tolerance_ms": res["tol_ms"],
           "sample_period_ms": res["dt"] * 1000.0, "n_spikes_matched": res["n_events"],
           "n_motion_cycles": res.get("n_cycles", 0)}
    if res["n_events"]:
        out["per_imu"] = {str(i): {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                                    for k, v in s.items()} for i, s in res["stats"].items()}
        out["max_spike_spread_ms"] = float(res["spread_ms"].max())
        out["median_spike_spread_ms"] = float(np.median(res["spread_ms"]))
    (outdir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def dominant_axis(df, ids, prefix, kind):
    stds = {ax: np.mean([df[f"{prefix}{i}_{kind}{ax}"].std() for i in ids]) for ax in "xyz"}
    return max(stds, key=stds.get)


def plot_overview(t, df, ids, prefix, sig, res, path, kind):
    gax = dominant_axis(df, ids, prefix, "g")
    aax = dominant_axis(df, ids, prefix, "a")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    panels = [(sig, f"{'Gyro magnitude' if kind == 'gyro' else 'Accel dynamic magnitude'} (used for spike detection)"),
              ({i: df[f"{prefix}{i}_g{gax}"].to_numpy() for i in ids}, f"Raw gyro {gax} (dominant gyro axis)"),
              ({i: df[f"{prefix}{i}_a{aax}"].to_numpy() for i in ids}, f"Raw accel {aax} (dominant accel axis)")]
    for ax, (data, title) in zip(axes, panels):
        for c, i in enumerate(ids):
            ax.plot(t, data[i], lw=1.3, color=PALETTE[c % len(PALETTE)], alpha=0.85, label=f"IMU {i}")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        if res["n_events"]:
            for cyc in range(res["n_cycles"]):
                i0, i1 = res["windows"][cyc]
                ax.axvspan(t[i0], t[min(i1, len(t)) - 1], color="0.5", alpha=0.10, lw=0)
    if res["n_events"]:
        for cyc in range(res["n_cycles"]):
            i0, i1 = res["windows"][cyc]
            axes[0].text((t[i0] + t[min(i1, len(t)) - 1]) / 2, 0.97, f"{cyc + 1}", ha="center", va="top",
                         fontsize=9, color="0.35", transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlabel("time (s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ids), frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("IMU overlay: do all sensors see the same motion at the same time?  "
                 "(grey bands / numbers = motion cycles)", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zoom(t, ids, sig, res, path):
    n = res["n_cycles"]
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), squeeze=False)
    for c in range(nrows * ncols):
        ax = axs[c // ncols][c % ncols]
        if c >= n:
            ax.axis("off")
            continue
        i0, i1 = res["windows"][c]
        for ci, i in enumerate(ids):
            ax.plot(t[i0:i1], sig[i][i0:i1], "-o", ms=2.8, lw=1.0, color=PALETTE[ci % len(PALETTE)],
                    alpha=0.85, label=f"IMU {i}")
        for tt in res["T"][res["cycle"] == c, res["rc"]]:
            ax.axvline(tt, color="0.5", ls=":", lw=0.9)
        ax.set_title(f"Cycle {c + 1}  ({t[i0]:.1f}-{t[min(i1, len(t)) - 1]:.1f} s)", fontsize=10)
        ax.grid(alpha=0.25)
        if c // ncols == nrows - 1:
            ax.set_xlabel("time (s)")
        if c % ncols == 0:
            ax.set_ylabel("signal")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ids), frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Zoom on every move/stop cycle (dots = actual samples; dotted line = reference spike)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lags(ids, res, path):
    ref, tol, dt = res["ref"], res["tol_ms"], res["dt"]
    others = [i for i in ids if i != ref]
    col = {i: PALETTE[ids.index(i) % len(PALETTE)] for i in ids}
    n = res["n_events"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    ax.axhspan(-tol, tol, color="0.5", alpha=0.12, lw=0, label=f"+/-1 sample ({tol:.0f} ms)")
    for k, i in enumerate(others):
        off = (k - (len(others) - 1) / 2) * 0.12
        ax.plot(np.arange(1, n + 1) + off, res["lag_ms"][:, ids.index(i)], "o", ms=5,
                color=col[i], label=f"IMU {i}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("spike number (x slightly offset per IMU)")
    ax.set_ylabel(f"lag vs IMU {ref} (ms)")
    ax.set_title("Lag of every spike", loc="left")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[0, 1]
    data = [res["lag_ms"][:, ids.index(i)] for i in others]
    ax.axhspan(-tol, tol, color="0.5", alpha=0.12, lw=0)
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True)
    for patch, i in zip(bp["boxes"], others):
        patch.set(facecolor=col[i], alpha=0.35)
    rng = np.random.default_rng(0)
    for k, (d, i) in enumerate(zip(data, others), start=1):
        ax.plot(k + rng.uniform(-0.15, 0.15, len(d)), d, "o", ms=3.5, color=col[i], alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(1, len(others) + 1))
    ax.set_xticklabels([f"IMU {i}" for i in others])
    ax.set_ylabel(f"lag vs IMU {ref} (ms)")
    ax.set_title("Lag distribution per IMU", loc="left")
    ax.grid(alpha=0.25)

    ax = axs[1, 0]
    for i in others:
        lags_s, c = res["xc_curves"][i]
        if lags_s is None:
            continue
        m = np.abs(lags_s) <= 0.4
        ax.plot(lags_s[m] * 1000, c[m], "-o", ms=3, lw=1.1, color=col[i],
                label=f"IMU {i}: {res['xc_whole'][i][0]:+.1f} ms")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("normalised cross-correlation")
    ax.set_title(f"Cross-correlation vs IMU {ref}", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[1, 1]
    ax.bar(np.arange(1, n + 1), res["spread_ms"], color="#0072B2", alpha=0.8)
    ax.axhline(tol, color="#D55E00", ls="--", lw=1.2, label=f"1 sample = {tol:.0f} ms")
    ax.set_xlabel("spike number")
    ax.set_ylabel("max - min spike time across IMUs (ms)")
    ax.set_title("Spread between the earliest and latest IMU", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Lag analysis  -  verdict: {res['verdict']}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# self-test with synthetic data of known lag
# --------------------------------------------------------------------------
def selftest(a) -> int:
    fs = 20.0
    t = np.arange(0, 40, 1 / fs)
    centers = [4.3, 4.7, 10.0, 10.45, 15.6, 16.15, 22.9, 23.4, 32.45, 32.8]
    true = {1: 0.0, 2: 0.040, 3: -0.030, 4: 0.100, 5: 0.0}   # seconds; +ve = later than IMU 1
    rng = np.random.default_rng(1)

    def bump(tt):
        return sum((1.0 + 0.07 * k) * np.exp(-0.5 * ((tt - c) / 0.08) ** 2) for k, c in enumerate(centers))

    sig = {i: bump(t - d) + rng.normal(0, 0.01, len(t)) for i, d in true.items()}
    a.ref, a.prominence, a.prom_frac, a.tol_ms = 1, 0.0, 0.25, 50.0
    res = analyse(t, sig, a)
    ok = True
    print("selftest: true lag vs recovered (ms)")
    for i, d in true.items():
        if i == 1:
            continue
        got_pk, got_xc = res["stats"][i]["median_ms"], res["stats"][i]["xcorr_ms"]
        good = abs(got_pk - d * 1000) < 15 and abs(got_xc - d * 1000) < 15
        ok &= good
        print(f"  IMU{i}: true {d * 1000:+6.1f}  peaks {got_pk:+6.1f}  xcorr {got_xc:+6.1f}  {'OK' if good else 'FAIL'}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Check synchronisation of multiple IMUs from spike lags.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("csv", nargs="?", help="input CSV file")
    p.add_argument("--outdir", default="sync_results", help="output folder")
    p.add_argument("--prefix", default="ESP32_IMU_", help="column prefix; columns are <prefix><n>_ax..gz")
    p.add_argument("--time-col", default="sys_time", help="time column (seconds)")
    p.add_argument("--ref", type=int, default=1, help="reference IMU number")
    p.add_argument("--signal", choices=["gyro", "accel"], default="gyro", help="signal used to find spikes")
    p.add_argument("--prominence", type=float, default=0.0, help="absolute spike prominence (0 = automatic)")
    p.add_argument("--prom-frac", type=float, default=0.25, help="automatic prominence = frac x median IMU max")
    p.add_argument("--min-dist", type=float, default=0.2, help="min time between spikes in one IMU (s)")
    p.add_argument("--match-window", type=float, default=0.2, help="max time difference to match a spike (s)")
    p.add_argument("--cycle-gap", type=float, default=2.0, help="gap (s) that separates motion cycles")
    p.add_argument("--max-lag", type=float, default=1.0, help="max lag searched, whole-record xcorr (s)")
    p.add_argument("--max-lag-cycle", type=float, default=0.5, help="max lag searched, per-cycle xcorr (s)")
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

    df, t, ids = load_data(csv_path, a.prefix, a.time_col)
    dts = np.diff(t)
    dt = float(np.median(dts))
    imu_cols = [c for c in df.columns if c.startswith(a.prefix)]

    log.info("=" * 78)
    log.info("IMU SYNCHRONISATION CHECK")
    log.info("=" * 78)
    log.info(f"file        : {csv_path.name}")
    log.info(f"rows        : {len(df)}   duration: {t[-1]:.2f} s")
    if "local_time" in df.columns:
        log.info(f"local time  : {df['local_time'].iloc[0]}  ->  {df['local_time'].iloc[-1]}")
    log.info(f"sample step : median {dt * 1000:.2f} ms ({1 / dt:.2f} Hz), min {dts.min() * 1000:.2f}, "
             f"max {dts.max() * 1000:.2f}, std {dts.std() * 1000:.3f} ms")
    log.info(f"gaps > 1.5x step: {int((dts > 1.5 * dt).sum())}")
    log.info(f"IMUs found  : {ids}   NaNs in IMU columns: {int(df[imu_cols].isna().sum().sum())}")
    log.info(f"signal used : {a.signal}")
    log.info(f"lag resolution: one sample = {dt * 1000:.1f} ms (sub-sample values are interpolated estimates)")

    sig = build_signals(df, ids, a.prefix, a.signal)
    res = analyse(t, sig, a)
    log_report(res)
    write_tables(res, outdir)
    write_json(res, outdir, {"file": csv_path.name, "rows": len(df), "duration_s": float(t[-1]),
                             "signal": a.signal, "imus": ids})
    if res["n_events"]:
        plot_overview(t, df, ids, a.prefix, sig, res, outdir / "fig1_overview.png", a.signal)
        plot_zoom(t, ids, sig, res, outdir / "fig2_zoom_cycles.png")
        plot_lags(ids, res, outdir / "fig3_lag_analysis.png")
    log.info(f"\nResults written to: {outdir.resolve()}")
    return 0 if res["verdict"] == "SYNCHRONISED" else 2


if __name__ == "__main__":
    sys.exit(main())
