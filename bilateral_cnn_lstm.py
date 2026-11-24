import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic(n_patients, per_class, n_classes, n_channels, length, seed):
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.3, 1.5, size=(n_classes, n_channels))
    scales[0] = 0.2
    X, y, patient, side = [], [], [], []
    for p in range(n_patients):
        for s in (0, 1):
            gain = 0.6 if s == 0 else 1.0
            for c in range(n_classes):
                for _ in range(per_class):
                    X.append(rng.normal(size=(length, n_channels)) * (0.1 + gain * scales[c]))
                    y.append(c)
                    patient.append(p)
                    side.append(s)
    return np.stack(X), np.array(y), np.array(patient), np.array(side)


def load_data(path):
    d = np.load(path, allow_pickle=False)
    X, y, patient, side = d["X"], d["y"], d["patient"], d["side"]
    if X.ndim != 3:
        raise ValueError("X must have shape (N, T, C)")
    if not (len(X) == len(y) == len(patient) == len(side)):
        raise ValueError("X, y, patient, side must have the same length")
    return X.astype(np.float64), y.astype(np.int64), patient, side.astype(np.int64)


def preprocess(X, patient, side, fs, low, high, order, norm):
    if high >= fs / 2:
        raise ValueError("high cutoff must be below fs / 2")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    X = sosfiltfilt(sos, X, axis=1)
    if norm == "window":
        mav = np.mean(np.abs(X), axis=1, keepdims=True)
        return (X / (mav + 1e-8)).astype(np.float32)
    out = np.empty_like(X)
    for p in np.unique(patient):
        for s in np.unique(side):
            idx = (patient == p) & (side == s)
            if idx.any():
                mav = np.mean(np.abs(X[idx]), axis=(0, 1), keepdims=True)
                out[idx] = X[idx] / (mav + 1e-8)
    return out.astype(np.float32)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.pad = k - 1
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class Net(nn.Module):
    def __init__(self, n_channels, n_classes, lstm_input):
        super().__init__()
        self.lstm_input = lstm_input
        self.conv1 = CausalConv(n_channels, 24, 6)
        self.ln1 = nn.LayerNorm(24)
        self.conv2 = CausalConv(24, 48, 6)
        self.ln2 = nn.LayerNorm(48)
        self.lstm1 = nn.LSTM(48, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = torch.relu(self.conv1(h)).transpose(1, 2)
        h = self.ln1(h).transpose(1, 2)
        h = torch.relu(self.conv2(h)).transpose(1, 2)
        h = self.ln2(h)
        if self.lstm_input == "pooled":
            h = h.mean(dim=1, keepdim=True)
        h, _ = self.lstm1(h)
        h = self.drop1(h)
        h, _ = self.lstm2(h)
        h = self.drop2(h)
        return self.fc(h[:, -1, :])


def train_model(X, y, n_classes, args, device):
    model = Net(X.shape[2], n_classes, args.lstm_input).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=args.batch,
        shuffle=True,
    )
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        sched.step()
    return model


def evaluate(model, X, y, n_classes, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i : i + 256]).float().to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    cm = confusion_matrix(y, p, labels=list(range(n_classes)))
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    sens = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
    spec = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1), np.nan)
    return {
        "acc": float(accuracy_score(y, p)),
        "sens": float(np.nanmean(sens)),
        "spec": float(np.nanmean(spec)),
        "f1": float(f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)),
    }


def summarize(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--modes", type=str, nargs="+", default=["paretic", "fused"])
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--low", type=float, default=20.0)
    ap.add_argument("--high", type=float, default=300.0)
    ap.add_argument("--filter_order", type=int, default=4)
    ap.add_argument("--norm", type=str, choices=["subject", "window"], default="subject")
    ap.add_argument("--lstm_input", type=str, choices=["pooled", "sequence"], default="pooled")
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--lr_step", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--test_patients", type=int, default=2)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        X, y, patient, side = make_synthetic(8, 6, 4, 4, 750, args.seed)
    elif args.data:
        X, y, patient, side = load_data(args.data)
    else:
        raise SystemExit("provide --data or --synthetic")

    if args.classes:
        keep = np.isin(y, args.classes)
        X, y, patient, side = X[keep], y[keep], patient[keep], side[keep]
        remap = {c: i for i, c in enumerate(sorted(args.classes))}
        y = np.array([remap[c] for c in y])

    n_classes = int(y.max()) + 1
    X = preprocess(X, patient, side, args.fs, args.low, args.high, args.filter_order, args.norm)

    patients = np.unique(patient)
    if args.test_patients >= len(patients):
        raise SystemExit("test_patients must be smaller than the number of patients")

    results = {m: [] for m in args.modes}
    for r in range(args.n_repeats):
        rng = np.random.default_rng(args.seed + r)
        test_ids = rng.choice(patients, size=args.test_patients, replace=False)
        is_test = np.isin(patient, test_ids)
        test_mask = is_test & (side == 0)
        for m in args.modes:
            train_mask = (~is_test) & ((side == 0) if m == "paretic" else np.ones_like(side, dtype=bool))
            torch.manual_seed(args.seed + r)
            model = train_model(X[train_mask], y[train_mask], n_classes, args, device)
            res = evaluate(model, X[test_mask], y[test_mask], n_classes, device)
            results[m].append(res)
            print(json.dumps({"repeat": r, "mode": m, **res}))

    summary = {m: summarize(results[m]) for m in args.modes}
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "runs": results, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
