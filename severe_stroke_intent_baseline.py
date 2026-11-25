import argparse
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfilt, welch
from scipy.stats import binom
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

EEG_NAMES = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz", "C4", "T8", "CP3", "CP4", "P3", "Pz", "P4", "Oz"]
HEMI = {"left": ["C3", "CP3", "P3"], "right": ["C4", "CP4", "P4"]}
NEIGHBORS = {
    "C3": ["F3", "CP3", "T7", "Cz"],
    "CP3": ["C3", "P3", "T7"],
    "P3": ["CP3", "Pz", "Oz"],
    "C4": ["F4", "CP4", "T8", "Cz"],
    "CP4": ["C4", "P4", "T8"],
    "P4": ["CP4", "Pz", "Oz"],
}
STRATEGIES = ["eeg_ipsi", "eeg_contra", "emg_involved", "emg_unrelated", "emg_all", "hybrid"]
T0 = -3.0
WIN_SEC = 1.0
TRAIN_STEP_SEC = 0.25
N_TRAIN_WIN = 5
SLIDE_STEP_SEC = 0.02


def make_synthetic(n_patients, n_blocks, per_block, fs, seed):
    rng = np.random.default_rng(seed)
    T = int(7 * fs)
    t = np.arange(T) / fs + T0
    move = (t >= 0.5).astype(float)
    eeg, emg, patient, block, side = [], [], [], [], []
    for p in range(n_patients):
        s = int(rng.integers(0, 2))
        for b in range(n_blocks):
            for _ in range(per_block):
                phase = rng.uniform(0, 2 * np.pi, size=(len(EEG_NAMES), 1))
                e = rng.normal(size=(len(EEG_NAMES), T))
                e = e + (1.0 - 0.6 * move) * np.sin(2 * np.pi * 10 * t + phase)
                m = rng.normal(size=(4, T)) * (1.0 + 0.8 * move)
                eeg.append(e)
                emg.append(m)
                patient.append(p)
                block.append(b)
                side.append(s)
    return {
        "eeg": np.stack(eeg),
        "emg": np.stack(emg),
        "eeg_names": np.array(EEG_NAMES),
        "patient": np.array(patient),
        "block": np.array(block),
        "side": np.array(side),
    }


def load_data(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in ["eeg", "emg", "eeg_names", "patient", "block", "side"]}
    n = len(out["eeg"])
    if not (len(out["emg"]) == len(out["patient"]) == len(out["block"]) == len(out["side"]) == n):
        raise ValueError("eeg, emg, patient, block, side must have the same number of trials")
    if out["eeg"].ndim != 3 or out["emg"].ndim != 3 or out["eeg"].shape[2] != out["emg"].shape[2]:
        raise ValueError("eeg and emg must be (N, C, T) with the same T")
    return out


def burg_batch(x, order):
    W, N = x.shape
    f = x[:, 1:].copy()
    b = x[:, :-1].copy()
    a = np.ones((W, 1))
    E = (x * x).sum(axis=1) / N
    for _ in range(order):
        k = -2.0 * (f * b).sum(axis=1) / ((f * f).sum(axis=1) + (b * b).sum(axis=1) + 1e-12)
        ae = np.concatenate([a, np.zeros((W, 1))], axis=1)
        a = ae + k[:, None] * ae[:, ::-1]
        E = E * (1.0 - k * k)
        fn = f + k[:, None] * b
        bn = b + k[:, None] * f
        f = fn[:, 1:]
        b = bn[:, :-1]
    return a, E


def make_band_basis(order, fs):
    mats = []
    k = np.arange(order + 1)[:, None]
    for lo, hi in [(7.0, 13.0), (14.0, 30.0)]:
        f = np.arange(lo, hi + 1e-9, 0.5)
        mats.append(np.exp(-2j * np.pi * f[None, :] * k / fs))
    return mats


def ar_band_features(windows, order, basis, fs):
    a, E = burg_batch(windows, order)
    cols = []
    for m in basis:
        A = a @ m
        psd = E[:, None] / (fs * np.abs(A) ** 2 + 1e-20)
        cols.append(psd.mean(axis=1))
    return np.stack(cols, axis=1)


def train_starts(first_start_sec, fs):
    return np.round((first_start_sec + np.arange(N_TRAIN_WIN) * TRAIN_STEP_SEC - T0) * fs).astype(int)


def eeg_block_features(lap, fs, order, basis, win, step):
    rest_idx = train_starts(-2.0, fs)
    move_idx = train_starts(1.0, fs)
    tr, sl = [], []
    for c in range(lap.shape[0]):
        w_tr = np.stack([lap[c, i : i + win] for i in np.concatenate([rest_idx, move_idx])])
        w_sl = sliding_window_view(lap[c], win)[::step]
        tr.append(ar_band_features(w_tr, order, basis, fs))
        sl.append(ar_band_features(w_sl, order, basis, fs))
    return np.concatenate(tr, axis=1), np.concatenate(sl, axis=1)


def emg_block_features(emg, fs, win, step):
    rest_idx = train_starts(-2.0, fs)
    move_idx = train_starts(1.0, fs)
    idx = np.concatenate([rest_idx, move_idx])
    tr, sl = [], []
    for c in range(emg.shape[0]):
        d = np.abs(np.diff(emg[c]))
        csum = np.concatenate([[0.0], np.cumsum(d)])
        tr.append(np.array([csum[i + win - 1] - csum[i] for i in idx]))
        starts = np.arange(0, emg.shape[1] - win + 1, step)
        sl.append(csum[starts + win - 1] - csum[starts])
    return np.stack(tr, axis=1), np.stack(sl, axis=1)


def laplacian(eeg, names, target):
    ti = names.index(target)
    ni = [names.index(n) for n in NEIGHBORS[target]]
    return eeg[ti] - eeg[ni].mean(axis=0)


def band_mean_power(sig, fs, lo, hi):
    f, p = welch(sig, fs=fs, nperseg=min(sig.shape[-1], 1000), axis=-1)
    m = (f >= lo) & (f <= hi)
    return float(p[..., m].mean())


def compute_trial(eeg, emg, names, side, args, basis, need):
    fs = args.fs
    win = int(WIN_SEC * fs)
    step = int(SLIDE_STEP_SEC * fs)
    ipsi_hemi = "left" if side == 1 else "right"
    contra_hemi = "right" if side == 1 else "left"
    out = {}
    if need["ipsi"]:
        lap = np.stack([laplacian(eeg, names, t) for t in HEMI[ipsi_hemi]])
        out["ipsi_tr"], out["ipsi_sl"] = eeg_block_features(lap, fs, args.ar_order, basis, win, step)
        out["delta"] = band_mean_power(lap, fs, 1.0, 4.0)
        out["gamma"] = band_mean_power(lap, fs, 30.0, 48.0)
    if need["contra"]:
        lap = np.stack([laplacian(eeg, names, t) for t in HEMI[contra_hemi]])
        out["contra_tr"], out["contra_sl"] = eeg_block_features(lap, fs, args.ar_order, basis, win, step)
    if need["emg"]:
        out["emg_tr"], out["emg_sl"] = emg_block_features(emg, fs, win, step)
        rest_end = int((0.0 - T0) * fs)
        out["rest_wl"] = float(np.abs(np.diff(emg[args.emg_involved, :rest_end], axis=1)).mean())
    return out


def compose(f, strategy, part, args):
    if strategy == "eeg_ipsi":
        return f["ipsi_" + part]
    if strategy == "eeg_contra":
        return f["contra_" + part]
    if strategy == "emg_involved":
        return f["emg_" + part][:, args.emg_involved]
    if strategy == "emg_unrelated":
        return f["emg_" + part][:, args.emg_unrelated]
    if strategy == "emg_all":
        return f["emg_" + part]
    return np.concatenate([f["ipsi_" + part], f["emg_" + part][:, args.emg_involved]], axis=1)


def rejected_flags(feats, need, use_reject):
    n = len(feats)
    bad = np.zeros(n, dtype=bool)
    if not use_reject:
        return bad
    keys = []
    if need["ipsi"]:
        keys += ["delta", "gamma"]
    if need["emg"]:
        keys += ["rest_wl"]
    for k in keys:
        v = np.array([f[k] for f in feats])
        bad |= v > v.mean() + 3.0 * v.std()
    return bad


def run_patient(feats, blocks, strategies, args, need, t_end):
    labels = np.array([0] * N_TRAIN_WIN + [1] * N_TRAIN_WIN)
    bad = rejected_flags(feats, need, not args.no_reject)
    rest_mask = (t_end >= -2.0 - 1e-9) & (t_end <= 0.0 + 1e-9)
    move_mask = (t_end >= 1.0 - 1e-9) & (t_end <= 4.0 + 1e-9)
    store = {s: {"rest": [], "move": []} for s in strategies}
    for b in np.unique(blocks):
        test_idx = np.where(blocks == b)[0]
        train_idx = np.where((blocks != b) & (~bad))[0]
        if len(train_idx) == 0:
            continue
        for s in strategies:
            Xtr = np.concatenate([compose(feats[i], s, "tr", args) for i in train_idx])
            ytr = np.tile(labels, len(train_idx))
            sc = StandardScaler().fit(Xtr)
            lda = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), ytr)
            for i in test_idx:
                p = lda.predict(sc.transform(compose(feats[i], s, "sl", args)))
                store[s]["rest"].append(p[rest_mask] == 0)
                store[s]["move"].append(p[move_mask] == 1)
    res = {}
    for s in strategies:
        if not store[s]["rest"]:
            continue
        ar = float(np.concatenate(store[s]["rest"]).mean())
        am = float(np.concatenate(store[s]["move"]).mean())
        n = len(store[s]["rest"])
        thr = float(binom.ppf(1.0 - args.alpha, n, 0.5) / n)
        acc = (ar + am) / 2.0
        res[s] = {"acc": acc, "acc_rest": ar, "acc_move": am, "n_trials": n, "chance_upper": thr, "above_chance": bool(acc > thr)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--strategies", type=str, nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--emg_involved", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--emg_unrelated", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--ar_order", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--no_reject", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        d = make_synthetic(4, 4, 5, args.fs, args.seed)
    elif args.data:
        d = load_data(args.data)
    else:
        raise SystemExit("provide --data or --synthetic")

    names = [str(n) for n in d["eeg_names"]]
    need = {
        "ipsi": any(s in ("eeg_ipsi", "hybrid") for s in args.strategies),
        "contra": "eeg_contra" in args.strategies,
        "emg": any(s.startswith("emg") or s == "hybrid" for s in args.strategies),
    }
    if need["ipsi"] or need["contra"]:
        for hemi in HEMI.values():
            for t in hemi:
                for n in [t] + NEIGHBORS[t]:
                    if n not in names:
                        raise SystemExit("missing EEG channel " + n)

    fs = args.fs
    eeg_sos = butter(4, [0.1, 48.0], btype="bandpass", fs=fs, output="sos")
    emg_sos = butter(4, 20.0, btype="highpass", fs=fs, output="sos")
    eeg = sosfilt(eeg_sos, d["eeg"].astype(np.float64), axis=-1)
    emg = sosfilt(emg_sos, d["emg"].astype(np.float64), axis=-1)

    T = eeg.shape[2]
    win = int(WIN_SEC * fs)
    step = int(SLIDE_STEP_SEC * fs)
    n_sl = len(range(0, T - win + 1, step))
    t_end = T0 + (np.arange(n_sl) * step + win) / fs

    basis = make_band_basis(args.ar_order, fs)
    patients = np.unique(d["patient"])
    results = {}
    for p in patients:
        idx = np.where(d["patient"] == p)[0]
        feats = [compute_trial(eeg[i], emg[i], names, int(d["side"][i]), args, basis, need) for i in idx]
        results[str(p)] = run_patient(feats, d["block"][idx], args.strategies, args, need, t_end)
        print(json.dumps({"patient": str(p), **{s: round(r["acc"], 4) for s, r in results[str(p)].items()}}))

    summary = {}
    for s in args.strategies:
        accs = np.array([results[k][s]["acc"] for k in results if s in results[k]])
        above = np.array([results[k][s]["above_chance"] for k in results if s in results[k]])
        summary[s] = {
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "frac_above_chance": float(above.mean()),
            "n_patients": int(len(accs)),
        }
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "per_patient": results, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
