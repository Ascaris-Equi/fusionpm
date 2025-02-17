# train_eval.py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve

from config import (
    BATCH_SIZE, EPOCHS, THRESHOLD, LR, USE_CUDA,
    MODEL_SAVE_DIR
)


def transfer(y_prob, threshold=0.5):
    """ 概率 -> 0/1 分类 """
    return np.array([1 if x > threshold else 0 for x in y_prob])


def performances(y_true, y_pred, y_prob, print_=True):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    try:
        mcc = ((tp*tn) - (fn*fp)) / np.sqrt(np.float64((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    except:
        mcc = np.nan

    sensitivity = tp / (tp+fn) if (tp+fn) != 0 else 0
    specificity = tn / (tn+fp) if (tn+fp) != 0 else 0
    recall = sensitivity
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) != 0 else 0

    rocauc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)

    if print_:
        print(f'tn = {tn}, fp = {fp}, fn = {fn}, tp = {tp}')
        print(f'y_pred: 0 = {Counter(y_pred)[0]} | 1 = {Counter(y_pred)[1]}')
        print(f'y_true: 0 = {Counter(y_true)[0]} | 1 = {Counter(y_true)[1]}')
        print(f'auc={rocauc:.4f}|sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={accuracy:.4f}|mcc={mcc:.4f}')
        print(f'precision={precision:.4f}|recall={recall:.4f}|f1={f1:.4f}|aupr={aupr:.4f}')

    return (rocauc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr)


def performances_to_pd(performances_list):
    import pandas as pd
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'sensitivity', 'specificity', 'precision', 'recall', 'aupr']
    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std']  = performances_pd.std(axis=0)
    return performances_pd


def train_step(model, train_loader, criterion, optimizer, fold=0, epoch=1, epochs=EPOCHS):
    model.train()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    y_true_train_list, y_prob_train_list = [], []
    loss_train_list = []
    time_train_ep = 0

    for train_pep_inputs, train_hla_inputs, train_labels in train_loader:
        train_pep_inputs = train_pep_inputs.to(device)
        train_hla_inputs = train_hla_inputs.to(device)
        train_labels = train_labels.to(device)

        t1 = time.time()
        train_outputs, _, _, _ = model(train_pep_inputs, train_hla_inputs)
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += (time.time() - t1)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].detach().cpu().numpy()

        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss.item())

    y_pred_train_list = transfer(y_prob_train_list, THRESHOLD)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

    print(
        f'Fold-{fold}****Train (Ep avg): Epoch-{epoch}/{epochs} '
        f'| Loss = {np.mean(loss_train_list):.4f} | Time = {time_train_ep:.4f} sec'
    )
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True)
    return ys_train, loss_train_list, metrics_train, time_train_ep


@torch.no_grad()
def eval_step(model, val_loader, criterion, fold=0, epoch=1, epochs=EPOCHS, desc='Val'):
    model.eval()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    loss_val_list = []
    y_true_val_list, y_prob_val_list = [], []

    for val_pep_inputs, val_hla_inputs, val_labels in val_loader:
        val_pep_inputs = val_pep_inputs.to(device)
        val_hla_inputs = val_hla_inputs.to(device)
        val_labels = val_labels.to(device)

        val_outputs, _, _, _ = model(val_pep_inputs, val_hla_inputs)
        val_loss = criterion(val_outputs, val_labels)

        y_true_val = val_labels.cpu().numpy()
        y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].detach().cpu().numpy()

        y_true_val_list.extend(y_true_val)
        y_prob_val_list.extend(y_prob_val)
        loss_val_list.append(val_loss.item())

    y_pred_val_list = transfer(y_prob_val_list, THRESHOLD)
    ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

    print(f'Fold-{fold} ****{desc}  Epoch-{epoch}/{epochs}: Loss = {np.mean(loss_val_list):.6f}')
    metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True)
    return ys_val, loss_val_list, metrics_val


def get_optimizer(model, lr=LR):
    return optim.Adam(model.parameters(), lr=lr)
