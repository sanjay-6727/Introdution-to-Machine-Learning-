#!/usr/bin/env python3
"""
Experiment 9 - Perceptron (PLA) vs Multilayer Perceptron (MLP) with hyperparameter tuning
ICS1512 Machine Learning Algorithms Laboratory, SSN College of Engineering

Usage
-----
  python run_experiment.py --data /path/to/english-handwritten-characters
        (folder containing english.csv and the Img/ directory of the Kaggle dataset)

Outputs (in ./out): figures (PNG) and results.json, which build_report.py turns into the LaTeX report.
Everything (PLA, MLP, optimisers, losses, metrics) is implemented from scratch with NumPy.
"""
import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

SEED = 42
IMG = 28
CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
NC = len(CHARS)
OUT = "out"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def prep_image(pil_img):
    """grayscale -> 28x28 -> ink=1 / background=0 -> per-image min-max scaling."""
    g = np.asarray(pil_img.convert("L").resize((IMG, IMG), Image.LANCZOS), dtype=np.float32) / 255.0
    g = 1.0 - g
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


def load_real(path):
    """Load the Kaggle 'English Handwritten Characters' set. `path` may be the extracted folder,
    a parent of it, or the downloaded .zip. Needs english.csv (columns image,label) + the Img/ folder."""
    import csv, glob, zipfile, tempfile
    if path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(); zipfile.ZipFile(path).extractall(tmp); path = tmp
    hits = glob.glob(os.path.join(path, "**", "english.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError("english.csv not found under " + path)
    root = os.path.dirname(hits[0])
    X, y = [], []
    with open(hits[0]) as f:
        for row in csv.DictReader(f):
            im = Image.open(os.path.join(root, row["image"])).convert("L")
            w, h = im.size; m = max(w, h)                       # pad to square (keep aspect ratio)
            sq = Image.new("L", (m, m), 255); sq.paste(im, ((m - w) // 2, (m - h) // 2))
            X.append(prep_image(sq).ravel())
            y.append(CHARS.index(row["label"]))
    return np.array(X, np.float32), np.array(y), "real"


# ----------------------------------------------------------------------------
# 2. Model A: Perceptron Learning Algorithm (one-vs-rest, step activation)
# ----------------------------------------------------------------------------
class PLA:
    """62 independent perceptrons (one per class). Step activation during training,
    update  w <- w + eta (y - y_hat) x  applied sample-by-sample (online). Prediction for the
    multi-class decision = argmax of the raw scores w_k.x."""

    def __init__(self, n_in, n_cls, eta=0.1, seed=SEED):
        self.W = np.zeros((n_in + 1, n_cls), np.float32)
        self.eta, self.rng = eta, np.random.default_rng(seed)

    @staticmethod
    def _aug(X):
        return np.hstack([X, np.ones((len(X), 1), np.float32)])

    def scores(self, X): return self._aug(X) @ self.W

    def predict(self, X): return self.scores(X).argmax(1)

    def fit(self, X, y, epochs=50, Xv=None, yv=None):
        Xa = self._aug(X)
        T = np.eye(self.W.shape[1], dtype=np.float32)[y]
        hist = {"train_err": [], "val_err": []}
        for ep in range(epochs):
            for i in self.rng.permutation(len(Xa)):
                yhat = (Xa[i] @ self.W >= 0).astype(np.float32)          # step activation
                self.W += self.eta * np.outer(Xa[i], T[i] - yhat)        # PLA update rule
            hist["train_err"].append(float((self.predict(X) != y).mean()))
            if Xv is not None:
                hist["val_err"].append(float((self.predict(Xv) != yv).mean()))
        return hist


# ----------------------------------------------------------------------------
# 3. Model B: MLP with back-propagation (from scratch)
# ----------------------------------------------------------------------------
def act_f(name, z):
    if name == "relu": return np.maximum(z, 0)
    if name == "tanh": return np.tanh(z)
    if name == "sigmoid": return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    raise ValueError(name)


def act_d(name, a, z):
    if name == "relu": return (z > 0).astype(z.dtype)
    if name == "tanh": return 1 - a * a
    if name == "sigmoid": return a * (1 - a)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    def __init__(self, n_in, hidden, n_out, activation="relu", loss="ce",
                 optimizer="adam", lr=1e-3, batch_size=64, l2=0.0, seed=SEED):
        self.act = activation
        self.loss_name = loss
        self.opt = optimizer
        self.lr = lr
        self.bs = batch_size
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [n_out]
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            s = np.sqrt(2.0 / a) if activation == "relu" else np.sqrt(1.0 / a)   # He / Xavier-style
            self.W.append((self.rng.standard_normal((a, b)) * s).astype(np.float32))
            self.b.append(np.zeros(b, np.float32))
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        A, Z = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            Z.append(z)
            A.append(softmax(z) if i == len(self.W) - 1 else act_f(self.act, z))
        return A, Z

    def predict_proba(self, X):
        return self.forward(X)[0][-1]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def loss(self, P, Y):
        if self.loss_name == "ce":
            return float(-np.mean(np.sum(Y * np.log(P + 1e-9), 1)))
        return float(np.mean(np.sum((P - Y) ** 2, 1)))

    def _step(self, Xb, Yb):
        A, Z = self.forward(Xb)
        P = A[-1]
        n = len(Xb)
        if self.loss_name == "ce":
            d = (P - Yb) / n                                    # softmax + cross-entropy
        else:
            g = 2 * (P - Yb) / n                                # MSE through softmax Jacobian
            d = P * (g - np.sum(g * P, 1, keepdims=True))
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = A[i].T @ d + self.l2 * self.W[i]
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * act_d(self.act, A[i], Z[i - 1])
        params, grads = self.W + self.b, gW + gb
        if self.opt in ("gd", "sgd"):
            for p, g in zip(params, grads): p -= self.lr * g
        elif self.opt == "adam":
            self.t += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            for k, (p, g) in enumerate(zip(params, grads)):
                self.m[k] = b1 * self.m[k] + (1 - b1) * g
                self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
                mh = self.m[k] / (1 - b1 ** self.t)
                vh = self.v[k] / (1 - b2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, y, epochs=60, Xv=None, yv=None, patience=None):
        Y = np.eye(self.W[-1].shape[1], dtype=np.float32)[y]
        Yv = None if Xv is None else np.eye(self.W[-1].shape[1], dtype=np.float32)[yv]
        h = {"train_loss": [], "val_loss": [], "train_err": [], "val_err": []}
        best, best_state, wait = 1e9, None, 0
        bs = len(X) if self.opt == "gd" else self.bs               # GD = full batch, SGD/Adam = mini-batch
        for ep in range(epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), bs):
                j = idx[s:s + bs]
                self._step(X[j], Y[j])
            P = self.predict_proba(X)
            h["train_loss"].append(self.loss(P, Y)); h["train_err"].append(float((P.argmax(1) != y).mean()))
            if Xv is not None:
                Pv = self.predict_proba(Xv)
                h["val_loss"].append(self.loss(Pv, Yv)); h["val_err"].append(float((Pv.argmax(1) != yv).mean()))
                if patience:
                    if h["val_err"][-1] < best - 1e-9:
                        best, wait = h["val_err"][-1], 0
                        best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                    else:
                        wait += 1
                        if wait >= patience: break
        if best_state is not None:
            self.W, self.b = best_state
        return h


# ----------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------
def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    return dict(acc=float((y == pred).mean()), prec=float(p), rec=float(r), f1=float(f),
                prec_w=float(pw), rec_w=float(rw), f1_w=float(fw))


def roc_micro_macro(y, S):
    Y = label_binarize(y, classes=list(range(NC)))
    fpr, tpr, _ = roc_curve(Y.ravel(), S.ravel())
    micro = (fpr, tpr, auc(fpr, tpr))
    fprs, tprs = [], []
    for k in range(NC):
        if Y[:, k].sum() == 0: continue
        f_, t_, _ = roc_curve(Y[:, k], S[:, k]); fprs.append(f_); tprs.append(t_)
    grid = np.linspace(0, 1, 500)
    mt = np.mean([np.interp(grid, f_, t_) for f_, t_ in zip(fprs, tprs)], 0)
    macro = (grid, mt, auc(grid, mt))
    return micro, macro


def plot_cm(y, pred, title, fn):
    cm = confusion_matrix(y, pred, labels=list(range(NC)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NC)); ax.set_yticks(range(NC))
    ax.set_xticklabels(CHARS, fontsize=5); ax.set_yticklabels(CHARS, fontsize=5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(fn, dpi=170); plt.close(fig)
    return cm


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    t0 = time.time()
    X, y, src = load_real(args.data)
    print("data source:", src, X.shape, "classes:", len(set(y)))

    # stratified 70/15/15 split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 0.1                        # standardise with train statistics only
    Xtr, Xva, Xte = [(a - mu) / sd for a in (Xtr, Xva, Xte)]
    print("split:", len(Xtr), len(Xva), len(Xte))
    R = dict(source=src, n_total=int(len(X)), n_train=int(len(Xtr)), n_val=int(len(Xva)), n_test=int(len(Xte)),
             img=IMG)

    # sample grid figure (raw, un-standardised)
    fig, axs = plt.subplots(4, 16, figsize=(12, 3.4))
    order = np.argsort(y, kind="stable")
    pick = [np.where(y == c)[0][0] for c in range(0, NC, 4)][:16]
    pick2 = [np.where(y == c)[0][1] for c in range(0, NC, 4)][:16]
    pick3 = [np.where(y == c)[0][2] for c in range(1, NC, 4)][:16]
    pick4 = [np.where(y == c)[0][3] for c in range(2, NC, 4)][:16]
    for a in axs.ravel(): a.axis("off")
    for r, pk in enumerate([pick, pick2, pick3, pick4]):
        for c, i in enumerate(pk):
            axs[r, c].imshow(X[i].reshape(IMG, IMG), cmap="gray_r"); axs[r, c].axis("off")
            axs[r, c].set_title(CHARS[y[i]], fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/samples.png", dpi=150); plt.close(fig)

    # class balance
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.bar(range(NC), np.bincount(y, minlength=NC)); ax.set_xticks(range(NC)); ax.set_xticklabels(CHARS, fontsize=6)
    ax.set_ylabel("images"); ax.set_title("Class distribution"); fig.tight_layout()
    fig.savefig(f"{OUT}/class_dist.png", dpi=150); plt.close(fig)

    # ---------------- Model A: PLA ----------------
    EP_PLA = 50
    pla_tune = {}
    for eta in [0.001, 0.01, 0.1, 1.0]:
        m = PLA(Xtr.shape[1], NC, eta=eta); h = m.fit(Xtr, ytr, EP_PLA, Xva, yva)
        pla_tune[str(eta)] = dict(val_acc=1 - h["val_err"][-1], best_val_acc=1 - min(h["val_err"]))
        print("PLA eta", eta, pla_tune[str(eta)])
    R["pla_tune"] = pla_tune
    eta_best = float(max(pla_tune, key=lambda k: pla_tune[k]["best_val_acc"]))
    pla = PLA(Xtr.shape[1], NC, eta=eta_best); hp = pla.fit(Xtr, ytr, EP_PLA, Xva, yva)
    R["pla"] = dict(eta=eta_best, epochs=EP_PLA, train=metrics(ytr, pla.predict(Xtr)),
                    test=metrics(yte, pla.predict(Xte)), hist=hp)
    print("PLA test", R["pla"]["test"])

    # ---------------- Model B: MLP tuning (coordinate search on validation accuracy) ----------------
    EP = 60
    base = dict(hidden=[128], activation="relu", loss="ce", optimizer="adam", lr=1e-3, batch_size=64, l2=0.0)
    log = []

    def run(cfg, epochs=EP):
        model = MLP(
            Xtr.shape[1], cfg["hidden"], NC,
            cfg["activation"], cfg["loss"], cfg["optimizer"],
            cfg["lr"], cfg["batch_size"], cfg["l2"]
        )

        start = time.time()
        hist = model.fit(Xtr, ytr, epochs, Xva, yva)

        train_acc = 1 - hist["train_err"][-1]
        val_acc = 1 - hist["val_err"][-1]

        return {
            "cfg": dict(cfg),
            "val_acc": float(val_acc),
            "train_acc": float(train_acc),
            "best_val_acc": float(1 - min(hist["val_err"])),
            "hist": hist,
            "secs": time.time() - start
        }

    def sweep(name, key_vals, cur):
        res = []
        for label, upd in key_vals:
            cfg = {**cur, **upd}
            r = run(cfg); r["stage"] = name; r["label"] = label; res.append(r); log.append(r)
            print(f"[{name}] {label:>22s} val={r['val_acc']:.3f} train={r['train_acc']:.3f} ({r['secs']:.0f}s)")
        best = max(res, key=lambda r: r["val_acc"])
        return best["cfg"], res

    cur = dict(base)
    # Stage 1: optimiser x learning rate
    grid = [("gd lr=%g" % lr, dict(optimizer="gd", lr=lr)) for lr in (0.1, 0.5, 1.0)] + \
           [("sgd lr=%g" % lr, dict(optimizer="sgd", lr=lr)) for lr in (0.01, 0.05, 0.1, 0.5)] + \
           [("adam lr=%g" % lr, dict(optimizer="adam", lr=lr)) for lr in (1e-4, 1e-3, 3e-3, 1e-2)]
    cur, s1 = sweep("optimizer_lr", grid, cur)
    # Stage 2: activation
    cur, s2 = sweep("activation", [(a, dict(activation=a)) for a in ("relu", "tanh", "sigmoid")], cur)
    # Stage 3: cost function
    cur, s3 = sweep("loss", [("cross-entropy", dict(loss="ce")), ("MSE", dict(loss="mse"))], cur)
    # Stage 4: batch size
    cur, s4 = sweep("batch", [(f"bs={b}", dict(batch_size=b)) for b in (16, 32, 64, 128, 256)], cur)
    # Stage 5: depth / width
    depth = [(f"{n}x128 lr={lr:g}", dict(hidden=[128] * n, lr=lr)) for n in (1, 2, 3, 4) for lr in (0.5, 0.1)] + \
            [("1x256", dict(hidden=[256])), ("1x512", dict(hidden=[512]))]
    cur, s5 = sweep("depth", depth, cur)
    # Stage 6: L2 regularisation
    cur, s6 = sweep("l2", [(f"l2={l:g}", dict(l2=l)) for l in (0.0, 1e-4, 1e-3, 1e-2)], cur)
    R["stages"] = {n: [dict(label=r["label"], val_acc=r["val_acc"], train_acc=r["train_acc"],
                            best_val_acc=r["best_val_acc"], secs=r["secs"],
                            ep_fit=next((i + 1 for i, e in enumerate(r["hist"]["train_err"]) if e == 0), None),
                            loss_ep10=r["hist"]["train_loss"][9], final_loss=r["hist"]["train_loss"][-1]) for r in s]
                   for n, s in dict(optimizer_lr=s1, activation=s2, loss=s3, batch=s4, depth=s5, l2=s6).items()}
    R["chosen"] = cur
    print("chosen:", cur)

    # ---------------- Final MLP (train with early stopping on validation) ----------------
    fm = MLP(Xtr.shape[1], cur["hidden"], NC, cur["activation"], cur["loss"], cur["optimizer"],
             cur["lr"], cur["batch_size"], cur["l2"])
    hf = fm.fit(Xtr, ytr, 150, Xva, yva, patience=20)
    R["mlp"] = dict(train=metrics(ytr, fm.predict(Xtr)), val=metrics(yva, fm.predict(Xva)),
                    test=metrics(yte, fm.predict(Xte)), hist=hf, epochs_run=len(hf["train_err"]))
    print("MLP test", R["mlp"]["test"])

    # untuned baseline (naive defaults) for "impact of tuning"
    naive = dict(hidden=[128], activation="sigmoid", loss="mse", optimizer="sgd", lr=0.01, batch_size=64, l2=0.0)
    rn = run(naive)
    nm = MLP(Xtr.shape[1], naive["hidden"], NC, naive["activation"], naive["loss"], naive["optimizer"],
             naive["lr"], naive["batch_size"], naive["l2"])
    hn = nm.fit(Xtr, ytr, EP, Xva, yva)
    R["naive"] = dict(cfg=naive, test=metrics(yte, nm.predict(Xte)), hist=hn)

    # overfitting demo: chosen config w/o L2 & no early stopping vs with (150 epochs)
    o_cfg = dict(cur, l2=0.0)
    om = MLP(Xtr.shape[1], o_cfg["hidden"], NC, o_cfg["activation"], o_cfg["loss"], o_cfg["optimizer"],
             o_cfg["lr"], o_cfg["batch_size"], 0.0)
    ho = om.fit(Xtr, ytr, 100, Xva, yva)
    R["overfit_demo"] = dict(hist=ho, test=metrics(yte, om.predict(Xte)))

    # ---------------- ROC + confusion matrices ----------------
    S_pla = pla.scores(Xte)
    P_mlp = fm.predict_proba(Xte)
    roc = {}
    for name, S in (("PLA", S_pla), ("MLP", P_mlp)):
        mi, ma = roc_micro_macro(yte, S)
        roc[name] = dict(micro_auc=float(mi[2]), macro_auc=float(ma[2]))
        plt.figure(figsize=(5, 4.4))
        plt.plot(mi[0], mi[1], label=f"micro-average (AUC={mi[2]:.3f})")
        plt.plot(ma[0], ma[1], label=f"macro-average (AUC={ma[2]:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"ROC - {name} (test)"); plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(f"{OUT}/roc_{name.lower()}.png", dpi=170); plt.close()
    R["roc"] = roc
    cm_pla = plot_cm(yte, pla.predict(Xte), "Confusion matrix - PLA (test)", f"{OUT}/cm_pla.png")
    cm_mlp = plot_cm(yte, fm.predict(Xte), "Confusion matrix - tuned MLP (test)", f"{OUT}/cm_mlp.png")

    # most confused pairs for MLP
    off = cm_mlp.copy(); np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:6], off.shape))[0]
    R["confused"] = [(CHARS[a], CHARS[b], int(off[a, b])) for a, b in pairs]
    # case-insensitive confusion share of MLP errors
    errs = fm.predict(Xte) != yte
    same_letter = sum(1 for t, p in zip(yte[errs], fm.predict(Xte)[errs])
                      if CHARS[t].lower() == CHARS[p].lower() or {CHARS[t], CHARS[p]} in ({"0", "O"}, {"0", "o"}, {"1", "l"}, {"1", "I"}, {"I", "l"}))
    R["confused_share"] = float(same_letter / max(errs.sum(), 1))
    per_cls = (cm_mlp.diagonal() / cm_mlp.sum(1).clip(1))
    R["worst_cls"] = [(CHARS[i], float(per_cls[i])) for i in np.argsort(per_cls)[:6]]
    R["best_cls"] = [(CHARS[i], float(per_cls[i])) for i in np.argsort(-per_cls)[:6]]

    # ---------------- Figures ----------------
    ep = np.arange(1, EP_PLA + 1)
    plt.figure(figsize=(5.4, 3.8))
    plt.plot(ep, hp["train_err"], label="train error"); plt.plot(ep, hp["val_err"], label="validation error")
    plt.xlabel("epoch"); plt.ylabel("misclassification rate"); plt.title("PLA convergence"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/pla_curve.png", dpi=170); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    e2 = np.arange(1, len(hf["train_loss"]) + 1)
    ax[0].plot(e2, hf["train_loss"], label="train"); ax[0].plot(e2, hf["val_loss"], label="validation")
    ax[0].set_title("Tuned MLP - cross-entropy loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(e2, hf["train_err"], label="train"); ax[1].plot(e2, hf["val_err"], label="validation")
    ax[1].set_title("Tuned MLP - error rate"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/mlp_curve.png", dpi=170); plt.close(fig)

    # optimiser convergence (each optimiser's best lr from stage 1)
    plt.figure(figsize=(5.6, 3.9))
    for opt in ("gd", "sgd", "adam"):
        rs = [r for r in s1 if r["cfg"]["optimizer"] == opt]; b = max(rs, key=lambda r: r["val_acc"])
        plt.plot(np.arange(1, EP + 1), b["hist"]["train_loss"], label=f"{opt} (lr={b['cfg']['lr']:g})")
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("training loss (log)"); plt.title("Optimiser convergence")
    plt.legend(); plt.grid(alpha=.3, which="both"); plt.tight_layout(); plt.savefig(f"{OUT}/opt_curve.png", dpi=170); plt.close()

    # tuning bars
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for a, (n, s) in zip(axs.ravel(), [("optimizer_lr", s1), ("activation", s2), ("loss", s3),
                                        ("batch", s4), ("depth", s5), ("l2", s6)]):
        a.bar(range(len(s)), [r["train_acc"] for r in s], width=.4, label="train", align="edge")
        a.bar(np.arange(len(s)) + .4, [r["val_acc"] for r in s], width=.4, label="val", align="edge")
        a.set_xticks(np.arange(len(s)) + .4); a.set_xticklabels([r["label"] for r in s], rotation=45, ha="right", fontsize=7)
        a.set_title(n); a.set_ylim(0, 1.02)
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/tuning.png", dpi=150); plt.close(fig)

    # depth vs generalisation gap
    plt.figure(figsize=(6.0, 4.2))
    lab = [r["label"] for r in s5]
    plt.plot(lab, [r["train_acc"] for r in s5], "o-", label="train"); plt.plot(lab, [r["val_acc"] for r in s5], "s-", label="val"); plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.ylabel("accuracy"); plt.title("Effect of depth / width"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/depth.png", dpi=170); plt.close()

    # overfitting demo
    plt.figure(figsize=(5.4, 3.8))
    plt.plot(ho["train_loss"], label="train loss"); plt.plot(ho["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.title("Generalisation gap (no regularisation)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/overfit.png", dpi=170); plt.close()

    # PLA vs MLP bar
    plt.figure(figsize=(5.4, 3.6))
    ks = ["acc", "prec", "rec", "f1"]
    x = np.arange(4)
    plt.bar(x - .2, [R["pla"]["test"][k] for k in ks], .4, label="PLA")
    plt.bar(x + .2, [R["mlp"]["test"][k] for k in ks], .4, label="MLP (tuned)")
    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1"]); plt.ylim(0, 1); plt.legend(); plt.title("Test metrics (macro)")
    plt.tight_layout(); plt.savefig(f"{OUT}/compare.png", dpi=170); plt.close()

    R["runtime_s"] = time.time() - t0
    json.dump(R, open(f"{OUT}/results.json", "w"))
    print("done in %.0fs" % R["runtime_s"])


if __name__ == "__main__":
    main()
