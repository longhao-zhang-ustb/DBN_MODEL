import torch.nn as nn
import torch.nn.init as init
import torch
from model_pth.attention import SelfAttentionLayer
import math
from model_pth.capsuleLayer import GRU_Layer, Caps_Layer, Dense_Layer, LSTM_Layer
from torch.autograd import Variable
# from torch_geometric.nn import GATConv
# capsule如果单纯做分类则不需要重构(reconstruction)
# 如果就用在分类里面，decoder用不到，不需要reconstruction

class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe:[1, 30, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class DSSMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(DSSMClassifier, self).__init__()
        self.emb_dim = 256
        self.d_model = 512
        self.nhead = 8
        
        # 通道1进行特征提取
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_dim, 1400),
            nn.BatchNorm1d(1400),
            nn.GELU(),
            nn.Linear(1400, 700),
            nn.BatchNorm1d(700),
            nn.GELU(),
            nn.Linear(700, self.emb_dim)
        )
        # self.gru_linear_bn = nn.BatchNorm1d(self.emb_dim)
        # self.gru_layer = GRU_Layer(1, self.emb_dim)
        # self.lstm_layer = LSTM_Layer(self.emb_dim, self.emb_dim)
        # self.xlstm_layer = xLSTM(self.emb_dim, head_size=self.emb_dim, num_heads=4, layers="msm", batch_first=True)
        # 【重要】初始化GRU权重操作，这一步非常关键，acc上升到0.98，如果用默认的uniform初始化则acc一直在0.5左右
        # self.gru_layer.init_weights()
        # self.lstm_layer.init_weights()
        
        # self.att_linear1 = nn.Linear(input_dim, 1024)
        # self.att_linear1_bn = nn.BatchNorm1d(1024)
        #####################GRU#############################
        self.att_layer = SelfAttentionLayer(self.emb_dim, n_heads=self.nhead, dropout=dropout, device="cpu")
        self.att_linear2 = nn.Linear(self.emb_dim, self.d_model)
        #####################XLSTM############################
        # self.att_layer = SelfAttentionLayer(1024, n_heads=self.nhead, dropout=dropout, device="cpu")
        # self.att_linear2 = nn.Linear(1024, self.d_model)
        
        init.xavier_uniform_(self.att_linear2.weight)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.caps_layer = Caps_Layer(self.d_model)
        self.dense_layer = Dense_Layer(output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # 设置残差连接
        ch1_x = self.mlp_layer(x)
        ch1_x = self.att_layer(ch1_x, ch1_x, ch1_x)
        ch1_x = self.att_linear2(ch1_x)
        ch1_x = self.dropout(ch1_x)
        x = self.caps_layer(ch1_x)
        x = self.dense_layer(x)
        return x