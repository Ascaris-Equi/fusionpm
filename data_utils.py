# data_utils.py
import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data

from config import PEP_MAX_LEN, HLA_MAX_LEN

class MyDataSet(Data.Dataset):
    """自定义数据集，与DataLoader配合使用"""
    def __init__(self, pep_inputs, hla_inputs, labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]


def make_data(data, vocab, pep_max_len=PEP_MAX_LEN, hla_max_len=HLA_MAX_LEN):
    """根据DataFrame，生成可供模型使用的tensor"""
    pep_inputs, hla_inputs, labels = [], [], []
    for pep, hla, label in zip(data.peptide, data.HLA_sequence, data.label):
        # 补齐长度
        pep = pep.ljust(pep_max_len, '-')
        hla = hla.ljust(hla_max_len, '-')
        # 转成索引
        pep_input = [[vocab[n] for n in pep]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)

    return (
        torch.LongTensor(pep_inputs),
        torch.LongTensor(hla_inputs),
        torch.LongTensor(labels)
    )


def get_data_loader(csv_path, vocab, batch_size=1024, shuffle=False):
    """
    根据给定csv文件路径，读取数据并返回dataloader
    参数 type_ / fold 等可以在外部自己拼接文件名
    """
    data = pd.read_csv(csv_path, index_col=0)
    pep_inputs, hla_inputs, labels = make_data(data, vocab)
    dataset = MyDataSet(pep_inputs, hla_inputs, labels)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data, loader
