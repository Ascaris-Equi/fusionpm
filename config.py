# config.py
import torch

# 随机种子设置
SEED = 19961231

# 数据相关
PEP_MAX_LEN = 15
HLA_MAX_LEN = 34
TGT_LEN = PEP_MAX_LEN + HLA_MAX_LEN

# 模型相关
D_MODEL = 64        # Embedding Size
D_FF = 512          # FeedForward dimension
D_K = 64            # dimension of K (Q)
D_V = 64            # dimension of V
N_LAYERS = 1        # number of Encoder/Decoder layer
N_HEADS = 9         # Multi-head attention中head数

# 训练相关
BATCH_SIZE = 1024
EPOCHS = 50
THRESHOLD = 0.5
LR = 1e-3           # 学习率
USE_CUDA = torch.cuda.is_available()

# 路径相关
COMMON_HLA_SEQ_PATH = 'Dataset/common_hla_sequence.csv'
VOCAB_DICT_PATH     = 'vocab_dict.npy'
MODEL_SAVE_DIR      = './model/pHLAIformer/'    # 权重保存目录