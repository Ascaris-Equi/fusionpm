# verify.py
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef, f1_score

def show(name, path):
    df = pd.read_csv(path)
    y = df["label"].astype(int).values
    p = df["score"].astype(float).values
    yh = (p > 0.5).astype(int)
    print(f"{name:25s} n={len(df):6d}  AUC={roc_auc_score(y,p):.4f}  AUPR={average_precision_score(y,p):.4f}  "
          f"ACC={accuracy_score(y,yh):.4f}  MCC={matthews_corrcoef(y,yh):.4f}  F1={f1_score(y,yh):.4f}")
    sb = (df["binder_class"]=="SB").sum(); wb = (df["binder_class"]=="WB").sum(); nb = (df["binder_class"]=="NB").sum()
    print(f"{'':25s} binder: SB={sb} WB={wb} NB={nb}")

show("INDEPENDENT (ensemble)", "_test_ind.csv")
show("INDEPENDENT (fast)    ", "_test_ind_fast.csv")
show("EXTERNAL    (ensemble)", "_test_ext.csv")