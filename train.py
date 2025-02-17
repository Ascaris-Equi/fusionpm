# main.py
import os
import torch
import numpy as np
import random
from tqdm import tqdm  # 如果需要
import pandas as pd

from config import (
    SEED, USE_CUDA,
    BATCH_SIZE, EPOCHS, THRESHOLD,
    MODEL_SAVE_DIR, VOCAB_DICT_PATH
)

from data_utils import get_data_loader
from model import Transformer
from train_eval import train_step, eval_step, get_optimizer, performances_to_pd


def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    # 0. 随机种子设置
    set_random_seed()

    # 1. 准备vocab字典
    vocab = np.load(VOCAB_DICT_PATH, allow_pickle=True).item()
    vocab_size = len(vocab)
    print("Vocab size =", vocab_size)

    # 2. 准备数据 DataLoader （这里你可以直接写你自己的路径）
    fold_idx = 0  # 示例
    train_csv_path = f"TransPHLA-AOMP/Dataset/train_data_fold{fold_idx}.csv"
    val_csv_path   = f"TransPHLA-AOMP/Dataset/val_data_fold{fold_idx}.csv"
    independent_csv_path = "TransPHLA-AOMP/Dataset/independent_set.csv"
    external_csv_path    = "TransPHLA-AOMP/Dataset/external_set.csv"

    _, train_loader       = get_data_loader(train_csv_path, vocab, batch_size=BATCH_SIZE, shuffle=True)
    _, val_loader         = get_data_loader(val_csv_path, vocab, batch_size=BATCH_SIZE, shuffle=False)
    _, independent_loader = get_data_loader(independent_csv_path, vocab, batch_size=BATCH_SIZE, shuffle=False)
    _, external_loader    = get_data_loader(external_csv_path, vocab, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化模型
    device = torch.device("cuda" if USE_CUDA else "cpu")
    model = Transformer(vocab_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)

    # 如果需要创建模型保存路径
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    save_path = os.path.join(MODEL_SAVE_DIR, f"model_fold{fold_idx}.pkl")

    # 4. 开始训练
    best_metric_avg = 0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        # 训练
        ys_train, loss_train_list, metrics_train, time_train_ep = train_step(
            model, train_loader, criterion, optimizer,
            fold=fold_idx, epoch=epoch, epochs=EPOCHS
        )

        # 验证
        ys_val, loss_val_list, metrics_val = eval_step(
            model, val_loader, criterion,
            fold=fold_idx, epoch=epoch, epochs=EPOCHS, desc="Val"
        )

        # 自行定义用何种指标来决定是否“最佳”
        # 这里示例：取val阶段(roc_auc + accuracy + mcc + f1) / 4 作为参考
        val_avg_metric = sum(metrics_val[:4]) / 4.0  # (roc_auc, acc, mcc, f1)的均值
        if val_avg_metric > best_metric_avg:
            best_metric_avg = val_avg_metric
            best_epoch = epoch
            # 保存
            torch.save(model.state_dict(), save_path)
            print(f"*** Saving best model at epoch {epoch}, best val_avg_metric={best_metric_avg:.4f}")

    print("Train finished.")
    print(f"Best epoch = {best_epoch}, best_metric_avg = {best_metric_avg:.4f}")
    
    # 5. 加载最优模型并评估
    if best_epoch > 0:
        model.load_state_dict(torch.load(save_path))
        model.eval()

        # 最终在 train、val、independent、external 上都跑一下
        print("==== Final Evaluation ====")
        print("Train set:")
        eval_step(model, train_loader, criterion, fold=fold_idx, epoch=best_epoch, desc="Train")
        print("Val set:")
        eval_step(model, val_loader, criterion, fold=fold_idx, epoch=best_epoch, desc="Val")
        print("Independent set:")
        eval_step(model, independent_loader, criterion, fold=fold_idx, epoch=best_epoch, desc="Independent")
        print("External set:")
        eval_step(model, external_loader, criterion, fold=fold_idx, epoch=best_epoch, desc="External")


if __name__ == "__main__":
    main()
