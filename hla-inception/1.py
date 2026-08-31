#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.py — HLA-Inception on 3 datasets, with HI_PRED_PATH fix + auto cache invalidation.
"""
import os, sys, re, time, json, shutil, subprocess, tarfile, threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT          = Path(__file__).resolve().parent
WORK          = ROOT / "hla_inception_run"
HLA_INC_DIR   = WORK / "HLA-Inception"
RESULTS_DIR   = WORK / "results"
LOG_PATH      = WORK / "all_results.txt"
N_WORKERS     = 12

TRANSPHLA_EXTRA   = ROOT / "transphla_run" / "extra_data"
FUSIONPM_DATA_DIR = ROOT / "transphla_run" / "fusionpm"
TRANSPHLA_REPO    = ROOT / "transphla_run" / "TransPHLA-AOMP"
HLA_INC_REPO  = "https://github.com/eawilson-CompBio/HLA-Inception.git"

WORK.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

LFS_HARDCODED = {
    "data/DefaultLengthWeights.txt":
        "8\t-2.45943714497759\n9\t0\n10\t-1.21752555050788\n11\t-1.93306550912998\n"
        "12\t-2.97942021274434\n13\t-3.72139908556076\n14\t-4.17771748086889\n"
        "15\t-4.35620863709117\n",
}

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f: f.write(line + "\n")

def run(cmd, cwd=None, check=True, capture=False, timeout=None, env=None):
    shown = cmd if isinstance(cmd, str) else " ".join(cmd)
    log(f"$ {shown}   (cwd={cwd or os.getcwd()})")
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, text=True, timeout=timeout,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None)
    if capture and res.stdout: log(res.stdout[-3000:])
    if check and res.returncode != 0:
        raise RuntimeError(f"cmd failed rc={res.returncode}: {shown}")
    return res

def run_streaming(cmd, cwd=None, env=None, timeout=None,
                  heartbeat_every=20, heartbeat_dirs=None, allow_nonzero=False):
    shown = cmd if isinstance(cmd, str) else " ".join(cmd)
    log(f"[stream] $ {shown}   (cwd={cwd or os.getcwd()})")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, universal_newlines=True,
                            shell=isinstance(cmd, str))
    start = time.time(); stop = threading.Event()
    def hb():
        while not stop.wait(heartbeat_every):
            elapsed = int(time.time() - start)
            parts = [f"elapsed={elapsed}s pid={proc.pid}"]
            for d in (heartbeat_dirs or []):
                try:
                    if Path(d).exists():
                        sz = subprocess.run(["du","-sh",str(d)],
                            capture_output=True, text=True, timeout=5).stdout.strip()
                        parts.append(f"{Path(d).name}={sz.split()[0] if sz else '?'}")
                except Exception: pass
            log("  [heartbeat] " + " | ".join(parts))
    t = threading.Thread(target=hb, daemon=True); t.start()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            print(f"  | {line}", flush=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"  | {line}\n")
            if timeout and (time.time()-start) > timeout:
                proc.kill(); raise TimeoutError(f"timed out {timeout}s")
        rc = proc.wait()
    finally:
        stop.set()
    log(f"[stream] rc={rc}  ({int(time.time()-start)}s)")
    if rc != 0 and not allow_nonzero:
        raise RuntimeError(f"cmd rc={rc}: {shown}")
    return rc

def is_lfs_pointer(p):
    p = Path(p)
    if not p.exists() or not p.is_file(): return False
    try: head = p.read_bytes()[:200]
    except Exception: return False
    return b"git-lfs.github.com/spec" in head

def is_real_file_for(path, must_extract_tar=False):
    p = Path(path)
    if not p.exists() or not p.is_file(): return False
    if is_lfs_pointer(p): return False
    if must_extract_tar:
        try:
            with tarfile.open(p, "r:*") as t: _ = t.getmembers()
            return True
        except Exception: return False
    return p.stat().st_size > 0

def check_tools():
    need = ["git", "git-lfs", "go"]
    miss = [t for t in need if shutil.which(t) is None]
    if miss:
        log(f"MISSING TOOLS: {miss}")
        log("sudo apt update && sudo apt install -y git git-lfs golang-go build-essential && git lfs install")
        sys.exit(2)
    for t in need:
        try:
            v = subprocess.run([t,"version"] if t=="go" else [t,"--version"],
                               capture_output=True, text=True).stdout.strip().splitlines()[0]
            log(f"  {t}: {v}")
        except Exception: pass
    try:
        import pandas, numpy, sklearn  # noqa
        log("  python deps: pandas/numpy/sklearn OK")
    except ImportError as e:
        log(f"MISSING python dep: {e}; pip install pandas numpy scikit-learn")
        sys.exit(2)
    try:
        subprocess.run(["go","env","-w","GOPROXY=https://goproxy.cn,direct"],
                       capture_output=True, text=True, timeout=20)
        subprocess.run(["go","env","-w","GOSUMDB=sum.golang.google.cn"],
                       capture_output=True, text=True, timeout=20)
    except Exception: pass

def clone_hla_inception():
    if not (HLA_INC_DIR / ".git").exists():
        env = os.environ.copy(); env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(["git","clone",HLA_INC_REPO,str(HLA_INC_DIR)], env=env, check=True)
    else:
        log(f"reuse {HLA_INC_DIR}")

def list_lfs_files():
    r = subprocess.run(["git","lfs","ls-files"],
                       cwd=str(HLA_INC_DIR), capture_output=True, text=True)
    files = []
    for line in r.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) >= 3: files.append(parts[2].strip())
    return files

def candidate_source_paths(rel_path):
    fname = Path(rel_path).name
    cands = [Path.cwd()/fname, ROOT/fname]
    for sub in ("downloads","Downloads","lfs","lfs_files","HLA-Inception-LFS"):
        cands += [ROOT/sub/fname, Path.cwd()/sub/fname]
    cands += [Path.home()/"Downloads"/fname, Path.cwd()/rel_path, ROOT/rel_path]
    seen, out = set(), []
    for c in cands:
        s = str(c.resolve())
        if s in seen: continue
        seen.add(s); out.append(c)
    return out

def restore_lfs_files():
    log("==== restore LFS files ====")
    lfs_list = list_lfs_files()
    if not lfs_list:
        for p in HLA_INC_DIR.rglob("*"):
            if p.is_file() and is_lfs_pointer(p):
                lfs_list.append(str(p.relative_to(HLA_INC_DIR)))
    log(f"LFS files tracked: {len(lfs_list)}")
    for f in lfs_list: log(f"  - {f}")
    missing = []
    for rel in lfs_list:
        target = HLA_INC_DIR / rel
        need_tar = target.suffix in (".tgz",".gz",".tar")
        if target.exists() and not is_lfs_pointer(target):
            if not need_tar or is_real_file_for(target, must_extract_tar=need_tar):
                log(f"  OK existing : {rel} ({target.stat().st_size} B)"); continue
        if rel in LFS_HARDCODED:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(LFS_HARDCODED[rel])
            log(f"  HARDCODED -> {rel}"); continue
        found = None
        for c in candidate_source_paths(rel):
            if is_real_file_for(c, must_extract_tar=need_tar): found = c; break
        if found:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, target)
            log(f"  IMPORT {found} -> {rel} ({target.stat().st_size} B)")
        else:
            missing.append(rel); log(f"  MISSING: {rel}")
    if missing: raise FileNotFoundError(f"LFS missing: {missing}")

def find_binary():
    names = ["hla-inception","HLA-Inception_pred","hla_inception","HLAInception",
             "HLA-Inception","hla-inception_pred","hi_pred","HI_pred"]
    for n in names:
        for sub in ("","src","bin","build"):
            p = HLA_INC_DIR/sub/n if sub else HLA_INC_DIR/n
            if p.exists() and p.is_file() and os.access(p, os.X_OK): return p
    for p in HLA_INC_DIR.rglob("*"):
        if ".git" in p.parts: continue
        if not (p.is_file() and os.access(p, os.X_OK)): continue
        if p.suffix in (".sh",".py",".pl",".rb",".txt",".md",".json",".tgz",".gz",".tar"): continue
        try: head = open(p,"rb").read(4)
        except Exception: continue
        if head[:4] == b"\x7fELF":
            return p
    return None

def install_hla_inception():
    bp = find_binary()
    if bp:
        log(f"binary already present: {bp}")
        return bp
    install_sh = HLA_INC_DIR / "install.sh"
    if not install_sh.exists(): raise FileNotFoundError("no install.sh")
    log("==== install.sh (streaming) ====")
    env = os.environ.copy()
    env.setdefault("GOPROXY","https://goproxy.cn,direct")
    env.setdefault("GOSUMDB","sum.golang.google.cn")
    env.setdefault("GO111MODULE","on")
    env["HI_PRED_PATH"] = str(HLA_INC_DIR)
    try:
        run_streaming(["bash","install.sh"], cwd=str(HLA_INC_DIR), env=env,
                      timeout=3600, allow_nonzero=True,
                      heartbeat_dirs=[str(HLA_INC_DIR/"data")])
    except TimeoutError as e:
        log(f"install.sh timeout: {e}")
    bp = find_binary()
    if not bp: raise RuntimeError("no binary after install.sh")
    try: os.chmod(bp, 0o755)
    except Exception: pass
    log(f"binary: {bp}")
    return bp

def parse_predictions(out_path):
    """对每行：取第一个 AA-only 长 8-15 的 token 作 peptide；取最后一个能转 float 的 token 作 score。"""
    res = {}
    with open(out_path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = re.split(r"[\t,;\s]+", line)
            if len(parts) < 2: continue
            pep = None
            for tok in parts:
                t = tok.strip().upper()
                if set(t) <= VALID_AA and 8 <= len(t) <= 15:
                    pep = t; break
            if pep is None: continue
            score = None
            for tok in parts[::-1]:
                try: score = float(tok); break
                except ValueError: continue
            if score is None: continue
            res[pep] = score
    return res

def smoke_test(bin_path):
    test_in = WORK / "smoke_peps.in"
    test_in.write_text("SIINFEKL\nGILGFVFTL\nNLVPMVATV\n")
    out = WORK / "smoke.out"
    if out.exists(): out.unlink()
    env = os.environ.copy()
    env["HI_PRED_PATH"] = str(HLA_INC_DIR)
    cmd = [str(bin_path),"-i",str(test_in),"-P","1","-a","A_02:01","-o",str(out)]
    log(f"smoke test: {' '.join(cmd)}")
    log(f"  HI_PRED_PATH={env['HI_PRED_PATH']}")
    r = subprocess.run(cmd, cwd=str(HLA_INC_DIR), text=True, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    log(f"smoke rc={r.returncode}")
    if r.stdout:
        for ln in r.stdout.splitlines()[:30]:
            log(f"  stdout> {ln}")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("smoke test failed (no file)")
    txt = out.read_text(errors="ignore")
    log(f"pred.out size = {out.stat().st_size}")
    log("pred.out content:")
    for ln in txt.splitlines()[:20]:
        log(f"  pred> {ln}")
    if not re.search(r"[ACDEFGHIKLMNPQRSTVWY]{8,15}", txt):
        raise RuntimeError("smoke: no peptide-like token in output")
    if not re.search(r"-?\d+\.\d+", txt):
        raise RuntimeError("smoke: no numeric score in output")
    n = len(parse_predictions(out))
    log(f"parse_predictions: {n} predictions")
    if n == 0:
        raise RuntimeError("smoke: parse_predictions returned 0 — format mismatch")

# ----- diagnose existing cache -----
def diagnose_cache():
    total = 0; valid = 0; sample = None
    bad_examples = []
    if not RESULTS_DIR.exists(): return 0, 0, None, []
    for tag_dir in RESULTS_DIR.iterdir():
        if not tag_dir.is_dir(): continue
        for sub in tag_dir.iterdir():
            if not sub.is_dir(): continue
            pred = sub / "pred.out"
            if not pred.exists(): continue
            total += 1
            try: txt = pred.read_text(errors="ignore")
            except Exception: continue
            has_pep = bool(re.search(r"[ACDEFGHIKLMNPQRSTVWY]{8,15}", txt))
            has_num = bool(re.search(r"-?\d+\.\d+", txt))
            if has_pep and has_num:
                valid += 1
                if sample is None: sample = (str(pred), txt[:1500])
            else:
                if len(bad_examples) < 3:
                    bad_examples.append((str(pred), txt[:200], pred.stat().st_size))
    return total, valid, sample, bad_examples

def force_clear_cache():
    n = 0
    if not RESULTS_DIR.exists(): return 0
    for tag_dir in RESULTS_DIR.iterdir():
        if not tag_dir.is_dir(): continue
        for sub in tag_dir.iterdir():
            if not sub.is_dir(): continue
            pred = sub / "pred.out"
            if pred.exists():
                pred.unlink(); n += 1
    return n

# ----- allele/peptide -----
ALLELE_RE = re.compile(r"^(?:HLA[-_])?([ABC])\*?(\d{2,4}):?(\d{0,3})$")
def to_hla_inc_allele(raw):
    if not isinstance(raw, str): return None
    s = raw.strip().upper().replace(" ","")
    m = ALLELE_RE.match(s)
    if not m: return None
    locus, f1, f2 = m.group(1), m.group(2), m.group(3)
    if f2 == "" and len(f1) >= 4: f2 = f1[2:4]; f1 = f1[:2]
    if not (f1 and f2): return None
    return f"{locus}_{f1.zfill(2)}:{f2.zfill(2)}"

def is_valid_peptide(p):
    return isinstance(p,str) and 8 <= len(p) <= 15 and set(p) <= VALID_AA

def load_seq_to_name_map():
    if not TRANSPHLA_REPO.exists(): return {}
    cands = list(TRANSPHLA_REPO.rglob("common_hla_sequence.csv"))
    if not cands: return {}
    import pandas as pd
    df = pd.read_csv(cands[0])
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("hla") or df.columns[0]
    seq_col  = cols.get("sequence") or cols.get("hla_sequence") or df.columns[-1]
    m = {}
    for _, r in df.iterrows():
        nm  = str(r[name_col]).strip()
        seq = str(r[seq_col]).strip().upper()
        if seq: m[seq] = nm
    return m

def find_fusionpm_csv():
    if not FUSIONPM_DATA_DIR.exists(): return None
    for n in ["test_result.csv","test_results.csv"]:
        p = FUSIONPM_DATA_DIR / n
        if p.exists() and p.stat().st_size > 100: return p
    return None

def prep_df(csv_path, tag, seq2name):
    import pandas as pd
    df = pd.read_csv(csv_path)
    log(f"[{tag}] cols={list(df.columns)} shape={df.shape}")
    log(f"[{tag}] head:\n{df.head(3).to_string()}")
    cols = {c.lower(): c for c in df.columns}
    pep_col = cols.get("peptide") or cols.get("peptides") or df.columns[0]
    hla_col = (cols.get("hla") or cols.get("allele") or cols.get("hla_name")
               or cols.get("mhc") or cols.get("mhc_allele"))
    seq_col = (cols.get("hla_sequence") or cols.get("hla_seq")
               or cols.get("mhc_sequence") or cols.get("pseudo_sequence"))
    label_col = (cols.get("label") or cols.get("y") or cols.get("target")
                 or cols.get("y_true") or cols.get("true_label"))
    sub = __import__("pandas").DataFrame()
    sub["peptide"] = df[pep_col].astype(str).str.upper().str.strip()
    if hla_col is not None:
        sub["hla_raw"] = df[hla_col].astype(str)
    elif seq_col is not None:
        sub["hla_raw"] = df[seq_col].astype(str).str.upper().str.strip().map(seq2name).fillna("")
    else:
        raise ValueError(f"{csv_path}: no HLA col")
    sub["hla_inc"] = sub["hla_raw"].map(to_hla_inc_allele)
    import pandas as pd
    sub["label"] = pd.to_numeric(df[label_col], errors="coerce") if label_col else float("nan")
    n0 = len(sub)
    valid = sub["peptide"].map(is_valid_peptide) & sub["hla_inc"].notna()
    log(f"[{tag}] rows in={n0} valid={int(valid.sum())} dropped={int((~valid).sum())}")
    return sub[valid].reset_index(drop=True)

def run_one_allele(args):
    allele, peps, sub_dir, bin_path, hla_inc_dir = args
    sub_dir = Path(sub_dir); sub_dir.mkdir(parents=True, exist_ok=True)
    in_p  = sub_dir / "peps.in"
    out_p = sub_dir / "pred.out"
    if out_p.exists() and out_p.stat().st_size > 0:
        return allele, str(out_p), "cached"
    in_p.write_text("\n".join(peps) + "\n")
    cmd = [str(bin_path),"-i",str(in_p),"-P","1","-a",allele,"-o",str(out_p)]
    env = os.environ.copy()
    env["HI_PRED_PATH"] = hla_inc_dir
    try:
        r = subprocess.run(cmd, cwd=str(hla_inc_dir), text=True, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10800)
        if r.returncode != 0 or not out_p.exists() or out_p.stat().st_size == 0:
            (sub_dir/"stderr.log").write_text(r.stdout or "")
            return allele, None, (r.stdout or "")[-400:]
    except subprocess.TimeoutExpired:
        return allele, None, "TIMEOUT"
    return allele, str(out_p), ""

def metrics(y, p):
    import numpy as np
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y = y[mask].astype(int); p = p[mask]
    out = {"n": int(len(y)), "n_pos": int((y==1).sum()), "n_neg": int((y==0).sum())}
    try:
        from sklearn.metrics import (roc_auc_score, average_precision_score,
                                     accuracy_score, f1_score, matthews_corrcoef)
        if out["n_pos"]>0 and out["n_neg"]>0:
            out["AUROC"] = float(roc_auc_score(y,p))
            out["AUPRC"] = float(average_precision_score(y,p))
        else:
            out["AUROC"] = float("nan"); out["AUPRC"] = float("nan")
        if p.size and p.max() > p.min():
            pn = (p - p.min()) / (p.max() - p.min())
        else: pn = p*0
        yhat = (pn >= 0.5).astype(int)
        out["ACC@0.5norm"] = float(accuracy_score(y,yhat))
        out["F1@0.5norm"]  = float(f1_score(y,yhat,zero_division=0))
        out["MCC@0.5norm"] = float(matthews_corrcoef(y,yhat))
        if out["n_pos"] > 0:
            order = np.argsort(-p); yhat_k = (y*0)
            yhat_k[order[:out["n_pos"]]] = 1
            out["ACC@topK"] = float(accuracy_score(y,yhat_k))
            out["F1@topK"]  = float(f1_score(y,yhat_k,zero_division=0))
            out["MCC@topK"] = float(matthews_corrcoef(y,yhat_k))
    except Exception as e:
        out["error"] = str(e)
    return out

def process_dataset(tag, csv_path, bin_path, seq2name):
    log(f"\n==== DATASET: {tag} ({csv_path}) ====")
    df = prep_df(csv_path, tag, seq2name)
    if len(df) == 0: return None
    work_dir = RESULTS_DIR / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    groups = list(df.groupby("hla_inc"))
    log(f"[{tag}] alleles={len(groups)} workers={N_WORKERS}")
    jobs = []
    for allele, sub in groups:
        sub_dir = work_dir / allele.replace(":","_")
        peps = sorted(set(sub["peptide"].tolist()))
        jobs.append((allele, peps, str(sub_dir), str(bin_path), str(HLA_INC_DIR)))
    preds = {}; errors = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one_allele, j): j[0] for j in jobs}
        done = 0; total = len(futs)
        for fut in as_completed(futs):
            a, op, err = fut.result(); done += 1
            if op is None:
                errors[a] = err
                log(f"[{tag}] ({done}/{total}) FAIL {a}: {err}")
            else:
                preds[a] = parse_predictions(op)
                log(f"[{tag}] ({done}/{total}) OK   {a}: {len(preds[a])} preds")
    df["hla_inception_score"] = float("nan")
    for i, r in df.iterrows():
        sc = preds.get(r["hla_inc"], {}).get(r["peptide"])
        if sc is not None: df.at[i,"hla_inception_score"] = sc
    raw_csv = work_dir / "raw_predictions.csv"
    df.to_csv(raw_csv, index=False)
    log(f"[{tag}] raw -> {raw_csv}")
    n_joined = int(df["hla_inception_score"].notna().sum())
    log(f"[{tag}] joined={n_joined} missing={len(df)-n_joined} failed_alleles={len(errors)}")
    overall, per_allele = None, {}
    if df["label"].notna().any():
        m = df[df["hla_inception_score"].notna() & df["label"].notna()]
        if len(m) > 0:
            overall = metrics(m["label"].tolist(), m["hla_inception_score"].tolist())
            for a, sub in m.groupby("hla_inc"):
                if len(sub) >= 10 and sub["label"].nunique() == 2:
                    per_allele[a] = metrics(sub["label"].tolist(),
                                            sub["hla_inception_score"].tolist())
    return {"tag":tag, "csv_path":str(csv_path), "raw_csv":str(raw_csv),
            "n_total":len(df), "n_joined":n_joined,
            "n_failed_alleles":len(errors), "errors":errors,
            "overall":overall, "per_allele":per_allele}

def write_report(results):
    L = ["="*78, "HLA-Inception inference on 3 datasets", "="*78]
    for tag, r in results.items():
        if r is None:
            L.append(f"\n--- {tag.upper()} --- SKIPPED"); continue
        L.append(f"\n--- {tag.upper()} ---")
        L.append(f"csv     = {r['csv_path']}")
        L.append(f"raw     = {r['raw_csv']}")
        L.append(f"n_total = {r['n_total']}  joined = {r['n_joined']}  failed_alleles = {r['n_failed_alleles']}")
        if r["overall"]:
            L.append("Overall:")
            for k,v in r["overall"].items(): L.append(f"  {k:<14}: {v}")
        if r["errors"]:
            L.append("Failed alleles (first 20):")
            for a, e in list(r["errors"].items())[:20]:
                L.append(f"  {a}: {e}")
        if r["per_allele"]:
            def _au(m):
                v = m.get("AUROC"); return -1 if v != v else v
            ranked = sorted(r["per_allele"].items(), key=lambda kv: _au(kv[1]), reverse=True)
            L.append(f"Per-allele (n={len(r['per_allele'])}, top 20 by AUROC):")
            for a, mm in ranked[:20]:
                L.append(f"  {a:>12}  n={mm.get('n'):>5}  "
                         f"AUROC={mm.get('AUROC'):.4f}  AUPRC={mm.get('AUPRC'):.4f}")
    L.append("\n"+"="*78)
    txt = "\n".join(L)
    with open(LOG_PATH,"a") as f: f.write("\n"+txt+"\n")
    print("\n"+txt)

def main():
    log("==== START HLA-Inception × 3 datasets ====")
    log(f"work = {WORK}  workers = {N_WORKERS}")
    check_tools()
    clone_hla_inception()
    restore_lfs_files()
    bin_path = install_hla_inception()

    # 关键：全局 HI_PRED_PATH
    os.environ["HI_PRED_PATH"] = str(HLA_INC_DIR)
    log(f"HI_PRED_PATH = {HLA_INC_DIR}")

    smoke_test(bin_path)

    # 诊断现有 cache，决定是否清掉
    total, valid, sample, bad = diagnose_cache()
    log(f"==== cache diagnose: total pred.out = {total}, valid (has AA+score) = {valid} ====")
    if sample:
        log(f"sample valid pred from: {sample[0]}")
        log("sample content (first 30 lines):")
        for ln in sample[1].splitlines()[:30]:
            log(f"  | {ln}")
    if bad:
        log("examples of INVALID pred.out:")
        for path, head, sz in bad:
            log(f"  bad: {path}  size={sz}")
            log(f"    head: {head!r}")
    if total > 0 and valid == 0:
        log(">>> all cached pred.out are EMPTY/INVALID -> clearing cache and re-running <<<")
        n = force_clear_cache()
        log(f"removed {n} stale pred.out files")
    elif total > 0 and valid < total:
        # 不强制清，但报告
        log(f">>> warning: {total-valid} pred.out look invalid; they will be REUSED. "
            "If you want to force re-run, manually delete them. <<<")

    DATASETS = {
        "independent": TRANSPHLA_EXTRA / "independent_set.csv",
        "external":    TRANSPHLA_EXTRA / "external_set.csv",
        "fusionpm":    find_fusionpm_csv(),
    }
    for tag, p in DATASETS.items():
        log(f"[{tag}] csv = {p}  exists={Path(p).exists() if p else False}")
    seq2name = load_seq_to_name_map()
    results = {}
    for tag in ("independent","external","fusionpm"):
        p = DATASETS[tag]
        if p is None or not Path(p).exists(): results[tag] = None; continue
        try:
            results[tag] = process_dataset(tag, Path(p), bin_path, seq2name)
        except Exception as e:
            log(f"[{tag}] FAIL: {e}")
            import traceback; log(traceback.format_exc())
            results[tag] = None
    write_report(results)
    with open(WORK/"metrics.json","w") as f:
        d = {k: None if r is None else {
            "overall": r["overall"], "n_total": r["n_total"],
            "n_joined": r["n_joined"], "n_failed_alleles": r["n_failed_alleles"]
        } for k, r in results.items()}
        json.dump(d, f, indent=2)
    log("==== DONE ====")

if __name__ == "__main__":
    main()