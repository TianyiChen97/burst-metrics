"""Re-implementation of MaxLab Live NetworkAnalysis_v2 ("ISI-N" burst detector) -> the 44 burst columns of
Run_22_23_24_42_43_AR_45_Network_Metrics.xlsx.   Self-contained: numpy, scipy, pandas, h5py only.

    from maxlab_network_metrics import network_metrics
    network_metrics("/path/data.raw.h5")                         # raw recording, all 6 wells  -> DataFrame (6 x 44)
    network_metrics("/path/.../internal_results.h5")             # MaxLab stored analysis (uses only its spike table)
    network_metrics((spike_times_s, spike_channels), well=3)     # plain arrays (one well)       -> Series (44)
    network_metrics(..., threshold=0.37)                         # override the automatic ISI-N threshold (seconds)

Inputs for the arrays case: spike_times in seconds (any order), spike_channels (any ids; only their number of distinct
values is used).  For raw files the preprocessing is: keep spikes on mapped channels, t=(frameno-first_routed_frame)/1e4,
drop t<0, stable sort.

STATUS (2026-11-17, vs the xlsx, 384 rows with a spike train): 349/384 rows exact on all 44 columns.  Known open cases:
  * 19 wells where MaxLab reports 0 bursts but this code finds some (rule for "no bursting" unknown),
  * 8 wells where the automatic threshold lands on the wrong grid point,
  * Run_45 wells that are non-bursting are NaN in the xlsx but 0.0 here.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

N = 300                      # ISI-N: number of consecutive spikes in a window
BIN, SIG = 0.01, 0.3         # network-activity curve: 10 ms bins, Gaussian sigma 0.3 s
GRID = np.logspace(-5, 1.5, 1000)   # candidate thresholds (s); MaxLab's threshold is always one of these
STATS = ["mean", "std", "CV", "median", "P10", "P90"]
GROUPS = ["SPB", "SPBch", "burstDur", "burstPeak", "IBI", "burstISI", "nonburstISI"]
COLUMNS = ["burstFrequency", "percSpikesinBursts"] + [f"{g}_{k}" for g in GROUPS for k in STATS]   # 44


# ----------------------------------------------------------------------------------------------- input
def load_raw_well(h5, w):
    """Detection spike train of well w (0-based) from an open data.raw.h5."""
    g = h5[f"wells/well{w:03d}/rec0000"]
    sp, mp = g["spikes"][:], g["settings/mapping"][:]
    f0 = int(h5[f"data_store/data{w:04d}/groups/routed/frame_nos"][0])
    sp = sp[np.isin(sp["channel"], mp["channel"])]
    t = (sp["frameno"] - f0) / 1e4
    k = t >= 0
    t, ch = t[k], sp["channel"][k]
    o = np.argsort(t, kind="stable")
    return t[o], ch[o]


def _trains(source, well):
    """Yield (well_number_1based, times, channels) for the supported inputs."""
    if isinstance(source, (tuple, list)) and len(source) == 2 and not isinstance(source[0], str):
        t, ch = np.asarray(source[0], float), np.asarray(source[1])
        o = np.argsort(t, kind="stable")
        yield (well or 1), t[o], ch[o]
        return
    path = str(source)
    wells = range(6) if well is None else [well - 1]
    if os.path.basename(path) == "internal_results.h5":
        for w in wells:
            r = pd.read_hdf(path, f"dataframes/{w}/relative_spike_times")
            yield w + 1, r.spike_times.values.astype(float), r.spike_channels.values
    else:
        import h5py
        with h5py.File(path, "r") as h5:
            for w in wells:
                t, ch = load_raw_well(h5, w)
                yield w + 1, t, ch


# ----------------------------------------------------------------------------------------------- threshold
def _kernel(sig):
    x = np.arange(-3 * sig, 3 * sig)
    return np.exp(-0.5 * (x / sig) ** 2)


def auto_threshold(s, n=N, sig0=5 + 1e-9, step=0.01, cap=50.0):
    """MaxLab's automatic threshold (s) from the histogram of N-spike window widths on the log grid.
    Adaptive Gaussian smoothing (sigma in grid bins) widened until <=2 peaks remain; with 2 peaks the threshold is the
    valley between them, otherwise the midpoint of the support of the sigma=5 curve.  Matches 524/544 stored wells."""
    if len(s) < 2 * n:
        return np.nan
    W = s[n - 1:] - s[:-(n - 1)]
    h = np.r_[np.histogram(W, GRID)[0], 0].astype(float)
    if h.sum() == 0:
        return np.nan
    y0 = np.convolve(h, _kernel(sig0), "same")
    y, pk, i, sg = y0, find_peaks(y0)[0], 0, sig0
    while len(pk) > 2:
        i += 1
        sg = sig0 + step * i
        if sg > cap:
            break
        y = np.convolve(h, _kernel(sg), "same")
        pk = find_peaks(y)[0]
    if len(pk) == 2:
        idx = pk[0] + int(np.argmin(y[pk[0]:pk[1] + 1]))
    else:
        nz = np.nonzero(y0)[0]
        idx = (nz[0] + nz[-1]) // 2
    return float(GRID[idx])


# ----------------------------------------------------------------------------------------------- bursts
def detect_bursts(s, thr, n=N):
    """Return [(i, j)] spike-index ranges of bursts.  Window c = spikes c..c+n-1, W[c]=s[c+n-1]-s[c]; qualifying windows
    (W<=thr) chain while c_next <= c_prev+n-1; burst = spikes [c_first+1, c_last+n-1] (note +1), start>=n-1, size>=n,
    stop<=L-n."""
    L, k = len(s), n - 1
    if L < 2 * n or not np.isfinite(thr):
        return []
    q = np.where(s[k:] - s[:-k] <= thr)[0]
    if len(q) == 0:
        return []
    chains, a, last = [], q[0], q[0]
    for c in q[1:]:
        if c <= last + k:
            last = c
        else:
            chains.append((a, last)); a = c; last = c
    chains.append((a, last))
    out = []
    for a, last in chains:
        start, end = max(a + 1, n - 1), last + k
        if end - start + 1 >= n:
            out.append((start, min(end, L - n)))
    return out


def _curve(s):
    """Network activity curve: 10 ms histogram, Gaussian (sigma 30 bins, window arange(-3sb,3sb)), 'same' convolution."""
    e = np.arange(0, s.max(), BIN)
    h = np.r_[np.histogram(s, e)[0], 0].astype(float)
    sb = SIG / BIN
    x = np.arange(-3 * sb, 3 * sb)
    g = np.exp(-0.5 * (x / sb) ** 2) / (sb * np.sqrt(2 * np.pi))
    return e, np.convolve(h, g, "same") / BIN


def _stats(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return dict.fromkeys(STATS, np.nan)
    return dict(mean=v.mean(), std=v.std(), CV=v.std() / v.mean(), median=np.median(v),
                P10=np.percentile(v, 10), P90=np.percentile(v, 90))


# ----------------------------------------------------------------------------------------------- metrics
def well_metrics(s, ch, thr, n=N):
    """44 metrics (+ n_bursts, threshold, isBursting) for one well.  s sorted seconds, ch channel ids.
    Returns (dict, per-burst DataFrame)."""
    L, nch, dur = len(s), len(np.unique(ch)), s[-1]
    det = detect_bursts(s, thr, n)
    grid = np.arange(0, s.max(), 0.01)
    snap = lambda t: grid[np.argmin(np.abs(grid - t))]        # nearest 10 ms grid point (ties as numpy decides)
    det = [(i, j) for i, j in det if snap(s[i]) < snap(s[j])]  # bursts collapsing to one grid point are dropped
    out = dict(n_bursts=len(det), threshold=thr, isBursting=len(det) >= 3)
    if len(det) < 3:                                          # non-bursting well: xlsx stores 0.0
        out.update({c: 0.0 for c in COLUMNS})
        return out, pd.DataFrame()

    nt, nf = _curve(s)
    ST = np.array([snap(s[i]) for i, j in det])
    SP = np.array([snap(s[j]) for i, j in det])
    rows, bid = [], np.full(L, -1)
    for q, ((i, j), st, sp) in enumerate(zip(det, ST, SP)):
        inside = (s > st) & (s < sp)                          # open snapped interval
        bid[inside] = q
        k0 = np.searchsorted(nt, st - 1e-9)
        k1 = max(np.searchsorted(nt, sp - 1e-9), k0 + 1)
        kp = k0 + np.argmax(nf[k0:k1])                        # peak time, half-open [start, stop)
        rows.append(dict(burstTime=nt[kp], start=st, stop=sp, SPB=int(inside.sum()), SPBch=inside.sum() / nch,
                         burstDur=np.round(s[j] - s[i], 2), burstPeak=nf[kp] / nch))
    B = pd.DataFrame(rows)
    d = np.diff(B.burstTime.values)
    B["IBI"] = np.r_[np.nan, d[1:], np.nan] if len(d) > 1 else np.nan

    bisi_all, nbisi, nb_cnt = [], [], np.zeros(len(det), int)
    segA, segB = SP, np.r_[ST[1:], s[-1]]                     # quiet periods; last one ends at the last spike (open)
    for c in np.unique(ch):
        idx = np.where(ch == c)[0]
        if len(idx) < 2:
            continue
        t, b = s[idx], bid[idx]
        dd = np.diff(t) * 1000                                # ms
        same = (b[:-1] >= 0) & (b[:-1] == b[1:])
        bisi_all.extend(dd[same])                             # within-burst ISIs of one channel
        inseg = (t[:, None] > segA[None, :]) & (t[:, None] < segB[None, :])   # exact float compare, strict
        sid = np.where(inseg.any(1), inseg.argmax(1), -1)
        keep = (sid[:-1] >= 0) & (sid[:-1] == sid[1:])
        nbisi.extend(dd[keep])
        nb_cnt += np.bincount(sid[:-1][keep], minlength=len(det))
    nbisi.extend([0.0] * int((nb_cnt == 0).sum()))            # a quiet period with no ISI contributes one 0.0

    out["burstFrequency"] = len(det) / dur
    out["percSpikesinBursts"] = 100 * B.SPB.sum() / L
    for g, col in [("SPB", "SPB"), ("SPBch", "SPBch"), ("burstDur", "burstDur"), ("burstPeak", "burstPeak")]:
        for k, v in _stats(np.round(B[col].values, 2)).items():   # well stats use the 2-dp rounded per-burst table
            out[f"{g}_{k}"] = v
    for g, v in [("IBI", B.IBI.dropna().values), ("burstISI", bisi_all), ("nonburstISI", nbisi)]:
        for k, x in _stats(v).items():
            out[f"{g}_{k}"] = x
    return out, B


def network_metrics(source, well=None, threshold=None, n=N, return_bursts=False):
    """See module docstring.  well: 1-based well number (default: all six for a file; 1 for arrays).
    threshold: seconds, or None for the automatic rule.  Returns a DataFrame indexed by well (or a Series for one well)."""
    rows, bursts = {}, {}
    for w, s, ch in _trains(source, well):
        thr = auto_threshold(s, n) if threshold is None else float(threshold)
        m, B = well_metrics(s, ch, thr, n)
        m["n_spikes"] = len(s)
        rows[w], bursts[w] = m, B
    df = pd.DataFrame(rows).T
    df.index.name = "well"
    cols = COLUMNS + ["n_bursts", "threshold", "isBursting", "n_spikes"]
    df = df[cols]
    res = df.iloc[0] if len(df) == 1 else df
    return (res, bursts) if return_bursts else res


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250, "display.max_rows", 100)
    print(network_metrics(sys.argv[1]).round(3).T.to_string())
