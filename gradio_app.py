# gradio_app.py
import gradio as gr
import torch
import torch.nn as nn
import numpy as np

from config import (
    USE_CUDA, D_MODEL, PEP_MAX_LEN, HLA_MAX_LEN,
    TGT_LEN, MODEL_SAVE_DIR, VOCAB_DICT_PATH, THRESHOLD
)
from model import Transformer
from data_utils import MyDataSet
from train_eval import transfer

device = torch.device("cuda" if USE_CUDA else "cpu")

# ====== 准备vocab及模型 =======
vocab = np.load(VOCAB_DICT_PATH, allow_pickle=True).item()
vocab_size = len(vocab)

model = Transformer(vocab_size).to(device)
# 你可以在此处指定你自己训练好的模型路径
model_path = f"{MODEL_SAVE_DIR}/model_fold0.pkl"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def encode_sequence(seq: str, max_len: int) -> torch.Tensor:
    """ 将输入序列(肽或HLA)根据 vocab 映射为数值并补齐到 max_len """
    seq = seq.strip()
    # 如果超长可自行处理，这里仅做简单截断
    if len(seq) > max_len:
        seq = seq[:max_len]
    # 补齐
    seq = seq.ljust(max_len, '-')
    # 转index
    seq_idx = [vocab.get(s, 0) for s in seq]
    return torch.LongTensor(seq_idx).unsqueeze(0)  # shape: [1, max_len]

def predict_peptide_hla(peptide: str, hla_seq: str) -> str:
    """
    用训练好的模型对输入的肽序列 + HLA序列进行预测
    返回预测概率和分类标签
    """
    pep_tensor = encode_sequence(peptide, PEP_MAX_LEN).to(device)
    hla_tensor = encode_sequence(hla_seq, HLA_MAX_LEN).to(device)
    
    with torch.no_grad():
        outputs, _, _, _ = model(pep_tensor, hla_tensor)
        prob = nn.Softmax(dim=1)(outputs)[:, 1].item()  # 阳性类的概率
        pred_label = 1 if prob > THRESHOLD else 0
    
    return f"Probability(1)={prob:.4f}, Predicted label={pred_label}"


# ====== Gradio界面搭建 =======
demo = gr.Interface(
    fn=predict_peptide_hla,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="输入肽序列(如: SIINFEKL)", label="Peptide"),
        gr.inputs.Textbox(lines=1, placeholder="输入HLA序列", label="HLA Sequence")
    ],
    outputs="text",
    title="pHLAIformer Prediction",
    description="输入抗原肽序列和HLA序列，点击下方按钮即可查看预测结果。"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
