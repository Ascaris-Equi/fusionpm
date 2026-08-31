# train.py
# Training over the five released split pairs. Saves only weights to ./weights/.
# Prints (does not save) TransPHLA-style independent + external test metrics per fold.
import os
import sys
import json
import time
import math
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
)

from model import (
    FusionPM, PEP_MAX_LEN, HLA_MAX_LEN, apply_mlm_mask,
)

# ---- defaults tuned for Ryzen 5600X + RTX 5090 ----
SEED = 19961231
N_FOLDS = 5
BATCH_SIZE = 4096
EPOCHS = 50
LR = 1e-3
THRESHOLD = 0.5
NUM_WORKERS = 6
MASK_RATE = 0.15
MLM_LOSS_WEIGHT = 0.1
LEGAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PAD = "-"

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
DATA_DIR = os.path.join(HERE, "dataset")
WEIGHTS_DIR = os.path.join(HERE, "weights")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_vocab():
    v = {PAD: 0}
    for i, a in enumerate(LEGAL_AA):
        v[a] = i + 1
    return v


def get_or_make_vocab():
    p = os.path.join(WEIGHTS_DIR, "vocab_dict.npy")
    if os.path.exists(p):
        v = np.load(p, allow_pickle=True).item()
        return v, p
    p2 = os.path.join(HERE, "vocab_dict.npy")
    if os.path.exists(p2):
        v = np.load(p2, allow_pickle=True).item()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        np.save(p, v)
        return v, p
    v = build_vocab()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    np.save(p, v)
    return v, p


def load_csv(path):
    # Tolerate both index_col=0 and standard CSVs.
    df = pd.read_csv(path, index_col=0) if os.path.exists(path) else None
    if df is None:
        raise FileNotFoundError(path)
    if not {"peptide", "HLA_sequence", "label"}.issubset(df.columns):
        df = pd.read_csv(path)
    return df


def encode_df(df, vocab):
    p_t, h_t, y_t = [], [], []
    for pep, hla, y in zip(df["peptide"], df["HLA_sequence"], df["label"]):
        ps = str(pep).strip().upper().ljust(PEP_MAX_LEN, PAD)[:PEP_MAX_LEN]
        hs = str(hla).strip().upper().ljust(HLA_MAX_LEN, PAD)[:HLA_MAX_LEN]
        try:
            p_t.append([vocab[c] for c in ps])
            h_t.append([vocab[c] for c in hs])
        except KeyError:
            continue
        y_t.append(int(y))
    return (torch.LongTensor(p_t), torch.LongTensor(h_t), torch.LongTensor(y_t))


class DS(Data.Dataset):
    def __init__(self, p, h, y):
        self.p, self.h, self.y = p, h, y
    def __len__(self): return self.p.shape[0]
    def __getitem__(self, i): return self.p[i], self.h[i], self.y[i]


def metrics(y_true, y_prob, threshold=THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob > threshold).astype(int)
    out = {}
    try: out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception: out["auc"] = float("nan")
    try: out["aupr"] = average_precision_score(y_true, y_prob)
    except Exception: out["aupr"] = float("nan")
    if set(y_true.tolist()).issubset({0, 1}) and len(set(y_true.tolist())) >= 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
        out["acc"] = (tp + tn) / max(1, tn + fp + fn + tp)
        d = math.sqrt(float(tp + fn) * (tn + fp) * (tp + fp) * (tn + fn)) or float("nan")
        out["mcc"] = ((tp * tn) - (fn * fp)) / d if d == d else float("nan")
        out["sens"] = tp / (tp + fn) if (tp + fn) else 0.0
        out["spec"] = tn / (tn + fp) if (tn + fp) else 0.0
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        out["prec"] = pr
        out["f1"] = 2 * pr * out["sens"] / (pr + out["sens"]) if (pr + out["sens"]) else 0.0
    else:
        for k in ("acc", "mcc", "sens", "spec", "prec", "f1"):
            out[k] = float("nan")
    return out


def make_loader(df, vocab, batch, shuffle, workers):
    p, h, y = encode_df(df, vocab)
    ds = DS(p, h, y)
    return Data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle,
        num_workers=workers, pin_memory=True,
        persistent_workers=(workers > 0), drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, dev, amp_dtype):
    model.eval()
    probs, trues = [], []
    for pep, hla, y in loader:
        pep = pep.to(dev, non_blocking=True)
        hla = hla.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=(dev.type == "cuda")):
            logits = model(pep, hla)
        p = nn.Softmax(dim=1)(logits.float())[:, 1].detach().cpu().numpy()
        probs.append(p); trues.append(y.numpy())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(trues)


def fmt(m):
    return ("auc=%.4f aupr=%.4f acc=%.4f mcc=%.4f f1=%.4f sens=%.4f spec=%.4f"
            % (m["auc"], m["aupr"], m["acc"], m["mcc"], m["f1"], m["sens"], m["spec"]))


def train_one_fold(fold, args, vocab, dev, amp_dtype):
    set_seed(SEED + fold)
    tr_csv = os.path.join(DATA_DIR, "train_data_fold%d.csv" % fold)
    va_csv = os.path.join(DATA_DIR, "val_data_fold%d.csv" % fold)
    ind_csv = os.path.join(DATA_DIR, "independent_set.csv")
    ext_csv = os.path.join(DATA_DIR, "external_set.csv")
    for p in (tr_csv, va_csv, ind_csv, ext_csv):
        if not os.path.exists(p):
            print("[fold %d] MISSING: %s" % (fold, p)); return None

    tr_df = load_csv(tr_csv); va_df = load_csv(va_csv)
    ind_df = load_csv(ind_csv); ext_df = load_csv(ext_csv)
    print("[fold %d] rows train=%d val=%d ind=%d ext=%d"
          % (fold, len(tr_df), len(va_df), len(ind_df), len(ext_df)))

    tr_loader = make_loader(tr_df, vocab, args.batch_size, True, args.num_workers)
    va_loader = make_loader(va_df, vocab, args.batch_size, False, args.num_workers)
    ind_loader = make_loader(ind_df, vocab, args.batch_size, False, args.num_workers)
    ext_loader = make_loader(ext_df, vocab, args.batch_size, False, args.num_workers)

    model = FusionPM(len(vocab)).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    main_crit = nn.CrossEntropyLoss()
    mlm_crit = nn.CrossEntropyLoss(ignore_index=-100)

    best_avg = -1.0
    best_state = None
    best_epoch = -1
    epoch_t = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses, mlm_losses = [], []
        for pep, hla, y in tr_loader:
            pep = pep.to(dev, non_blocking=True)
            hla = hla.to(dev, non_blocking=True)
            y   = y.to(dev, non_blocking=True)

            # MLM mask peptide (used as input + supervised target on masked positions)
            masked_pep, mask = apply_mlm_mask(pep, mask_rate=args.mask_rate)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype,
                                enabled=(dev.type == "cuda")):
                logits, mlm_logits = model(masked_pep, hla, mlm=True)
                loss_main = main_crit(logits, y)
                if mask.any():
                    targets = pep.clone()
                    targets[~mask] = -100
                    loss_mlm = mlm_crit(mlm_logits.reshape(-1, mlm_logits.shape[-1]),
                                        targets.reshape(-1))
                else:
                    loss_mlm = torch.zeros((), device=dev)
                loss = loss_main + args.mlm_w * loss_mlm

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss_main.item())
            mlm_losses.append(loss_mlm.item())

        val_p, val_t = predict(model, va_loader, dev, amp_dtype)
        vm = metrics(val_t, val_p)
        avg4 = float(np.mean([vm["auc"], vm["acc"], vm["mcc"], vm["f1"]]))
        print("[fold %d] ep %02d  loss=%.4f mlm=%.4f | val %s avg4=%.4f  (%.1fs)"
              % (fold, ep, float(np.mean(losses)), float(np.mean(mlm_losses)),
                 fmt(vm), avg4, time.time() - t0))

        if avg4 > best_avg:
            best_avg = avg4
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print("[fold %d] best ep=%d val_avg4=%.4f  total %.1f min"
          % (fold, best_epoch, best_avg, (time.time() - epoch_t) / 60))

    # restore best, evaluate on independent + external (PRINT ONLY)
    model.load_state_dict(best_state)
    ind_p, ind_t = predict(model, ind_loader, dev, amp_dtype)
    ext_p, ext_t = predict(model, ext_loader, dev, amp_dtype)
    im = metrics(ind_t, ind_p); em = metrics(ext_t, ext_p)
    print("[fold %d] INDEPENDENT  %s" % (fold, fmt(im)))
    print("[fold %d] EXTERNAL     %s" % (fold, fmt(em)))

    # save weights only
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out_w = os.path.join(WEIGHTS_DIR, "model_fold%d.pkl" % fold)
    torch.save(best_state, out_w)
    print("[fold %d] saved -> %s" % (fold, out_w))

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "val_avg4": best_avg,
        "val": vm,
        "independent": im,
        "external": em,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default=None,
                    help="comma list, e.g. 0,1,2 ; default all 0..4")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--mask_rate", type=float, default=MASK_RATE)
    ap.add_argument("--mlm_w", type=float, default=MLM_LOSS_WEIGHT)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--force", action="store_true",
                    help="re-train folds even if weights exist")
    args = ap.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    # bf16 on 5090 / Ampere+; fp16 fallback elsewhere.
    if dev.type == "cuda":
        major = torch.cuda.get_device_capability(0)[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.bfloat16

    print("=" * 60)
    print("[ENV] python=%s torch=%s cuda=%s" %
          (sys.version.split()[0], torch.__version__, torch.cuda.is_available()))
    if dev.type == "cuda":
        print("[ENV] GPU=%s  compute=%s  vram=%.1fGB" %
              (torch.cuda.get_device_name(0),
               torch.cuda.get_device_capability(0),
               torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    print("[CFG] dev=%s amp=%s batch=%d workers=%d epochs=%d lr=%g mask_rate=%.2f mlm_w=%.2f"
          % (dev, amp_dtype, args.batch_size, args.num_workers,
             args.epochs, args.lr, args.mask_rate, args.mlm_w))

    if not os.path.isdir(DATA_DIR):
        print("[FATAL] missing %s" % DATA_DIR); sys.exit(2)
    vocab, vp = get_or_make_vocab()
    print("[VOCAB] %s size=%d" % (vp, len(vocab)))

    folds = list(range(N_FOLDS)) if args.folds is None else \
            [int(x) for x in args.folds.split(",") if x.strip() != ""]
    print("[PLAN] folds=%s" % folds)

    summary = []
    t_all = time.time()
    for fold in folds:
        out_w = os.path.join(WEIGHTS_DIR, "model_fold%d.pkl" % fold)
        if (not args.force) and os.path.exists(out_w):
            print("[fold %d] SKIP (weight exists: %s)" % (fold, out_w))
            continue
        info = train_one_fold(fold, args, vocab, dev, amp_dtype)
        if info is not None:
            summary.append(info)

    print("\n=== SUMMARY ===")
    # find best fold by val_avg4 among newly-trained folds; if none new, scan disk
    best_fold, best_score = None, -1.0
    for s in summary:
        print("fold %d: best_ep=%d val_avg4=%.4f | IND %s | EXT %s"
              % (s["fold"], s["best_epoch"], s["val_avg4"],
                 fmt(s["independent"]), fmt(s["external"])))
        if s["val_avg4"] > best_score:
            best_score = s["val_avg4"]; best_fold = s["fold"]

    if best_fold is None:
        # fallback: pick whatever exists on disk; record fold 0 if file 0 exists
        for fi in range(N_FOLDS):
            if os.path.exists(os.path.join(WEIGHTS_DIR, "model_fold%d.pkl" % fi)):
                best_fold = fi; best_score = float("nan"); break

    if best_fold is not None:
        with open(os.path.join(WEIGHTS_DIR, "best_fold.json"), "w") as f:
            json.dump({"best_fold": int(best_fold),
                       "val_avg4": float(best_score)}, f, indent=2)
        print("[BEST FOLD] %d  (val_avg4=%.4f) -> weights/best_fold.json"
              % (best_fold, best_score))
    print("[DONE] total %.1f min" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
