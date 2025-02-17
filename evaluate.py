# evaluate.py

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    USE_CUDA, BATCH_SIZE, THRESHOLD, MODEL_SAVE_DIR,
    VOCAB_DICT_PATH
)
from data_utils import get_data_loader
from model import Transformer
from train_eval import performances, transfer

def load_model(model_path: str):
    """ 加载训练好的模型并返回 """
    device = torch.device("cuda" if USE_CUDA else "cpu")
    vocab = np.load(VOCAB_DICT_PATH, allow_pickle=True).item()
    model = Transformer(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab

@torch.no_grad()
def evaluate_csv(model, vocab, csv_path: str, dataset_name: str, fold: str = "NA"):
    """
    对单个CSV文件进行评估，返回 (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr)。
    model      : 加载好的模型
    vocab      : token->id 字典
    csv_path   : 数据文件路径
    dataset_name: 数据集名称(如 "Independent", "External", "Train", "Validation"...)
    fold       : 该文件对应的 fold 编号（如果没有可用 'NA'）
    """
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # 读取数据
    df, data_loader = get_data_loader(
        csv_path=csv_path,
        vocab=vocab,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    y_true_list = []
    y_prob_list = []

    for pep_inputs, hla_inputs, labels in data_loader:
        pep_inputs = pep_inputs.to(device)
        hla_inputs = hla_inputs.to(device)
        labels = labels.to(device)

        outputs, _, _, _ = model(pep_inputs, hla_inputs)
        probs = nn.Softmax(dim=1)(outputs)[:, 1].cpu().numpy()

        y_true_list.extend(labels.cpu().numpy())
        y_prob_list.extend(probs)

    y_pred_list = transfer(y_prob_list, THRESHOLD)
    # 计算各项指标
    rocauc, accuracy, mcc, f1_score, sensitivity, specificity, precision, recall, aupr = performances(
        y_true_list, y_pred_list, y_prob_list, print_=False
    )

    return {
        "fold": fold,
        "roc_auc": rocauc,
        "accuracy": accuracy,
        "mcc": mcc,
        "f1": f1_score,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "aupr": aupr,
        "dataset": dataset_name
    }

def main_evaluate(dataset_dir="TransPHLA-AOMP/Dataset"):
    """
    遍历指定目录下所有CSV文件，对它们进行评估，并汇总输出结果。
    - dataset_dir: CSV 文件所在目录
    """
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # 加载 vocab
    vocab = np.load(VOCAB_DICT_PATH, allow_pickle=True).item()
    vocab_size = len(vocab)

    # 在此可一次性加载你最想用的模型，如默认使用 fold0 的模型；
    # 或者你也可以根据CSV名中的 fold 去加载对应的 'model_fold{fold}.pkl'
    # 这里示例：如果文件名包含 foldX，就加载 model_foldX.pkl，否则默认加载 model_fold0.pkl
    # 当然也可以根据你自己的逻辑进行修改
    def get_model_for_fold(fold_id):
        model_path = os.path.join(MODEL_SAVE_DIR, f"model_fold{fold_id}.pkl")
        if not os.path.exists(model_path):
            # 如果对应fold的权重不存在，就使用 fold0 的
            model_path = os.path.join(MODEL_SAVE_DIR, "model_fold0.pkl")
        model_, vocab_ = load_model(model_path)
        return model_, vocab_

    # 用于汇总评估结果
    results = []

    # 遍历目录下所有CSV
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    csv_files.sort()  # 按名字排序，方便输出

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # 解析 fold 编号
        # 例如 train_data_fold0.csv -> fold=0
        fold_search = re.search(r"fold(\d+)", filename)
        fold_id = fold_search.group(1) if fold_search else "0"  # 如果没找到，则默认为0
        
        # 判断数据集名称(也可根据实际情况自行修改)
        if "independent" in filename.lower():
            dataset_name = "Independent"
        elif "external" in filename.lower():
            dataset_name = "External"
        elif "train" in filename.lower():
            dataset_name = "Train"
        elif "val" in filename.lower():
            dataset_name = "Validation"
        elif "test" in filename.lower():
            dataset_name = "Test"
        else:
            dataset_name = "Unknown"

        # 加载对应fold的模型
        model_, vocab_ = get_model_for_fold(fold_id)
        
        # 评估
        metrics_dict = evaluate_csv(
            model=model_, vocab=vocab_,
            csv_path=csv_file,
            dataset_name=dataset_name,
            fold=fold_id
        )
        results.append(metrics_dict)

    # 转成 DataFrame
    df_res = pd.DataFrame(results)

    # 排列列顺序，仅仅是让输出更美观
    df_res = df_res[
        ["fold", "roc_auc", "accuracy", "mcc", "f1",
         "sensitivity", "specificity", "precision", "recall", "aupr", "dataset"]
    ]

    # 按 dataset 分组
    final_rows = []
    row_index = 0
    for dataset_name, group in df_res.groupby("dataset"):
        # 先把每条fold结果输出
        # 并为每条记录标注一个行号
        for i in range(len(group)):
            row = group.iloc[i].copy()
            # DataFrame 里 fold 可能是字符串，转一下
            fold_str = row["fold"]
            # 生成你的自定义表格行（示例：与上面给出的表类似）
            # row_index, fold, roc_auc, accuracy, ...
            final_rows.append([
                row_index,
                fold_str,
                row["roc_auc"],
                row["accuracy"],
                row["mcc"],
                row["f1"],
                row["sensitivity"],
                row["specificity"],
                row["precision"],
                row["recall"],
                row["aupr"],
                dataset_name
            ])
            row_index += 1

        # group做均值、方差
        mean_ = group.mean(numeric_only=True)
        std_ = group.std(numeric_only=True)

        # mean 行
        final_rows.append([
            row_index,
            "mean",
            mean_["roc_auc"],
            mean_["accuracy"],
            mean_["mcc"],
            mean_["f1"],
            mean_["sensitivity"],
            mean_["specificity"],
            mean_["precision"],
            mean_["recall"],
            mean_["aupr"],
            dataset_name
        ])
        row_index += 1

        # std 行
        final_rows.append([
            row_index,
            "std",
            std_["roc_auc"],
            std_["accuracy"],
            std_["mcc"],
            std_["f1"],
            std_["sensitivity"],
            std_["specificity"],
            std_["precision"],
            std_["recall"],
            std_["aupr"],
            dataset_name
        ])
        row_index += 1

    # 整理最终输出
    final_df = pd.DataFrame(
        final_rows,
        columns=[
            "index", "fold", "roc_auc", "accuracy", "mcc", "f1", "sensitivity",
            "specificity", "precision", "recall", "aupr", "dataset"
        ]
    )
    
    # 打印并可视化
    print("===== Evaluation Results =====")
    print(final_df)

    # 也可以存成CSV
    output_csv_path = "evaluation_result.csv"
    final_df.to_csv(output_csv_path, index=False)
    print(f"汇总结果已保存到: {output_csv_path}")


if __name__ == "__main__":
    main_evaluate()
