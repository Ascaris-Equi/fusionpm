# infer.py
# Default: ensemble across ALL weights in ./weights/.
# --fast: load ONLY the best fold (per weights/best_fold.json).
# Outputs CSV with score, IC50_nM (NetMHCpan-style), binder_class, rank, ...
import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data

from model import FusionPM, PEP_MAX_LEN, HLA_MAX_LEN

DEFAULT_BATCH = 4096
LEGAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
PAD = "-"

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
WEIGHTS_DIR = os.path.join(HERE, "weights")
DATA_DIR = os.path.join(HERE, "dataset")


def build_vocab():
    v = {PAD: 0}
    for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY"):
        v[a] = i + 1
    return v


def load_vocab():
    for p in (os.path.join(WEIGHTS_DIR, "vocab_dict.npy"),
              os.path.join(HERE, "vocab_dict.npy")):
        if os.path.exists(p):
            return np.load(p, allow_pickle=True).item(), p
    return build_vocab(), "(built-in)"


def list_weights(fast):
    if fast:
        bj = os.path.join(WEIGHTS_DIR, "best_fold.json")
        if not os.path.exists(bj):
            print("[FATAL] --fast requires weights/best_fold.json (train.py writes it)")
            sys.exit(2)
        bi = json.load(open(bj))["best_fold"]
        p = os.path.join(WEIGHTS_DIR, "model_fold%d.pkl" % bi)
        if not os.path.exists(p):
            print("[FATAL] best_fold=%d but %s missing" % (bi, p)); sys.exit(2)
        return [p]
    return sorted(glob.glob(os.path.join(WEIGHTS_DIR, "model_fold*.pkl")))


def validate(pep, hla):
    if not isinstance(pep, str) or not isinstance(hla, str):
        return "non-string"
    p = pep.strip().upper(); h = hla.strip().upper()
    if not (8 <= len(p) <= 14): return "bad-pep-len(%d)" % len(p)
    if len(h) == 0 or len(h) > HLA_MAX_LEN: return "bad-hla-len(%d)" % len(h)
    bp = sorted(set(a for a in p if a not in LEGAL_AA))
    if bp: return "bad-pep-aa(%s)" % "".join(bp)
    bh = sorted(set(a for a in h if a not in LEGAL_AA))
    if bh: return "bad-hla-aa(%s)" % "".join(bh)
    return "ok"


def encode(pep, hla, vocab):
    p = pep.strip().upper().ljust(PEP_MAX_LEN, PAD)[:PEP_MAX_LEN]
    h = hla.strip().upper().ljust(HLA_MAX_LEN, PAD)[:HLA_MAX_LEN]
    return [vocab[c] for c in p], [vocab[c] for c in h]


def prob_to_ic50(p):
    arr = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return np.power(50000.0, 1.0 - arr)


def ic50_class(ic50):
    out = []
    for v in np.asarray(ic50, dtype=float):
        if v != v: out.append("NA")
        elif v < 50.0: out.append("SB")
        elif v < 500.0: out.append("WB")
        else: out.append("NB")
    return out


def load_hla_table():
    p = os.path.join(DATA_DIR, "common_hla.csv")
    if not os.path.exists(p): return None
    try:
        df = pd.read_csv(p)
        # try to autodetect columns
        cmap = {c.lower(): c for c in df.columns}
        a_col = cmap.get("hla") or cmap.get("allele")
        s_col = cmap.get("hla_sequence") or cmap.get("sequence") or cmap.get("pseudo_sequence")
        if a_col and s_col:
            d = dict(zip(df[a_col].astype(str), df[s_col].astype(str)))
            return d
    except Exception as e:
        print("[WARN] cannot parse common_hla.csv: %s" % e)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input CSV")
    ap.add_argument("--output", default=None, help="output CSV path")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=0,
                    help="keep only top-K peptides per HLA_sequence (0=all)")
    ap.add_argument("--fast", action="store_true",
                    help="single-fold mode (uses weights/best_fold.json)")
    ap.add_argument("--dry_run", action="store_true",
                    help="validate input, no inference")
    args = ap.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda":
        major = torch.cuda.get_device_capability(0)[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        amp_dtype = torch.bfloat16

    print("[ENV] torch=%s cuda=%s dev=%s amp=%s"
          % (torch.__version__, torch.cuda.is_available(), dev, amp_dtype))

    vocab, vp = load_vocab()
    print("[VOCAB] %s size=%d" % (vp, len(vocab)))
    if PAD not in vocab:
        print("[FATAL] vocab missing pad token"); sys.exit(2)

    wpaths = list_weights(args.fast)
    print("[WEIGHTS] %d file(s) (%s mode)"
          % (len(wpaths), "fast" if args.fast else "ensemble"))
    for w in wpaths: print("        - %s" % os.path.basename(w))
    if not wpaths:
        print("[FATAL] no model weights in %s" % WEIGHTS_DIR); sys.exit(2)

    if not os.path.exists(args.input):
        print("[FATAL] input not found: %s" % args.input); sys.exit(2)
    df = pd.read_csv(args.input)
    cmap = {c.lower(): c for c in df.columns}
    pep_col = cmap.get("peptide")
    hla_col = cmap.get("hla_sequence")
    allele_col = cmap.get("hla") or cmap.get("allele")
    id_col = cmap.get("id")
    print("[INFO] rows=%d cols=%s" % (len(df), list(df.columns)))

    # allow lookup HLA allele -> sequence via dataset/common_hla.csv
    if hla_col is None and allele_col is not None:
        table = load_hla_table()
        if table:
            df["_HLA_sequence"] = df[allele_col].astype(str).map(table)
            hla_col = "_HLA_sequence"
            n_unk = df[hla_col].isna().sum()
            print("[INFO] mapped HLA allele -> sequence via common_hla.csv "
                  "(%d unmapped)" % n_unk)
    if pep_col is None or hla_col is None:
        print("[FATAL] need columns: peptide and (HLA_sequence OR HLA)")
        sys.exit(2)

    status = [validate("" if pd.isna(p) else str(p),
                       "" if pd.isna(h) else str(h))
              for p, h in zip(df[pep_col], df[hla_col])]
    ok = np.array([s == "ok" for s in status])
    print("[INFO] valid %d/%d" % (int(ok.sum()), len(df)))

    if args.dry_run:
        out = df.copy(); out["status"] = status
        op = args.output or (args.input + ".dryrun.csv")
        out.to_csv(op, index=False); print("[DRY] -> %s" % op); return

    score = np.full(len(df), np.nan)
    nm = np.zeros(len(df), dtype=int)
    idx = np.where(ok)[0]

    if len(idx) > 0:
        pl, hl = [], []
        for i in idx:
            a, b = encode(str(df[pep_col].iloc[i]),
                          str(df[hla_col].iloc[i]), vocab)
            pl.append(a); hl.append(b)
        ds = Data.TensorDataset(torch.LongTensor(pl), torch.LongTensor(hl))
        loader = Data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=(dev.type == "cuda"))
        folds = []; n_used = 0
        for wp in wpaths:
            try:
                sd = torch.load(wp, map_location=dev, weights_only=False)
                if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
                m = FusionPM(len(vocab)).to(dev)
                miss, unexp = m.load_state_dict(sd, strict=False)
                if miss or unexp:
                    print("[WARN] %s miss=%d unexp=%d"
                          % (os.path.basename(wp), len(miss), len(unexp)))
                m.eval()
            except Exception as e:
                print("[WARN] %s failed to load: %s" % (os.path.basename(wp), e))
                continue
            t0 = time.time(); ps = []
            with torch.no_grad():
                for pep, hla in loader:
                    pep = pep.to(dev, non_blocking=True)
                    hla = hla.to(dev, non_blocking=True)
                    with torch.autocast(device_type=dev.type, dtype=amp_dtype,
                                        enabled=(dev.type == "cuda")):
                        logits = m(pep, hla)
                    ps.append(nn.Softmax(dim=1)(logits.float())[:, 1]
                              .detach().cpu().numpy())
            if ps:
                folds.append(np.concatenate(ps)); n_used += 1
                print("[INFER] %s (%.1fs)" %
                      (os.path.basename(wp), time.time() - t0))
            del m
            if dev.type == "cuda": torch.cuda.empty_cache()
        if n_used == 0:
            print("[FATAL] no fold usable"); sys.exit(3)
        score[idx] = np.mean(np.stack(folds, 0), 0)
        nm[idx] = n_used

    ic50 = np.where(np.isnan(score), np.nan,
                    prob_to_ic50(np.where(np.isnan(score), 0.5, score)))
    binder = ic50_class(ic50)

    out = df.copy()
    if id_col is None: out.insert(0, "id", np.arange(len(df)))
    out["score"] = score
    out["IC50_nM"] = ic50
    out["binder_class"] = binder
    out["pred_label"] = pd.array(
        [pd.NA if (v != v) else int(v > args.threshold) for v in score],
        dtype="Int64")
    out["n_models"] = nm
    out["status"] = status
    out["rank"] = pd.array([pd.NA] * len(out), dtype="Int64")
    for _, g in out.loc[ok].groupby(hla_col):
        out.loc[g.index, "rank"] = g["score"].rank(method="min", ascending=False).astype(int)
    if args.top_k and args.top_k > 0:
        out = out.loc[out["rank"].notna() & (out["rank"] <= args.top_k)].copy()
    # drop internal helper col
    if "_HLA_sequence" in out.columns:
        out = out.drop(columns=["_HLA_sequence"])

    op = args.output or (args.input + ".pred.csv")
    out.to_csv(op, index=False)
    print("[SAVE] -> %s" % op)
    for k, v in pd.Series(status).value_counts().items():
        print("        %s: %d" % (k, int(v)))


if __name__ == "__main__":
    main()