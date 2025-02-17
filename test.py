# test.py
import os
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

def test_model(csv_path, model_path, output_csv="test_result.csv"):
    """
    在给定的csv文件上做推理，并保存预测结果。
    csv_path: str, 待测试CSV文件路径
    model_path: str, 训练好的模型文件路径
    output_csv: str, 保存测试结果的CSV文件路径
    """
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    # 1. 准备Vocab
    vocab = np.load(VOCAB_DICT_PATH, allow_pickle=True).item()
    vocab_size = len(vocab)

    # 2. 构建数据Loader
    data_df, data_loader = get_data_loader(
        csv_path=csv_path,
        vocab=vocab,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 3. 初始化并加载模型
    model = Transformer(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. 推理
    y_true_list = []
    y_prob_list = []
    pep_list = []
    hla_list = []

    with torch.no_grad():
        for pep_inputs, hla_inputs, labels in data_loader:
            pep_inputs = pep_inputs.to(device)
            hla_inputs = hla_inputs.to(device)
            labels = labels.to(device)

            outputs, _, _, _ = model(pep_inputs, hla_inputs)
            probs = nn.Softmax(dim=1)(outputs)[:, 1].detach().cpu().numpy()

            y_true_list.extend(labels.cpu().numpy())
            y_prob_list.extend(probs)
    
    # 5. 将结果写到DataFrame（便于保存）
    y_pred_list = transfer(y_prob_list, THRESHOLD)
    data_df = data_df.reset_index(drop=True)

    # 在原有DataFrame上追加列
    data_df["pred_prob"] = y_prob_list
    data_df["pred_label"] = y_pred_list

    # 6. 计算性能指标并打印
    metrics = performances(y_true_list, y_pred_list, y_prob_list, print_=True)

    # 7. 保存到CSV
    data_df.to_csv(output_csv, index=False)
    print(f"测试结果已保存到：{output_csv}")


if __name__ == "__main__":
    # 示例：指定待测试集CSV文件路径与模型文件路径
    fold_idx = 0
    test_csv_path = "TransPHLA-AOMP/Dataset/independent_set.csv"  # 你自己的测试集路径
    model_path = os.path.join(MODEL_SAVE_DIR, f"model_fold{fold_idx}.pkl")
    
    # 输出保存结果
    output_csv = "test_result.csv"
    
    # 调用测试函数
    test_model(test_csv_path, model_path, output_csv)