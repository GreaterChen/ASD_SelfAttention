import logging

from SelfAttention import *
from requirements import *
from args import *
from utils import *


class Structure(nn.Module):
    def __init__(self):
        super(Structure, self).__init__()

        self.middle_size = 50
        # 展开、降维模块
        self.desc_fc = nn.Sequential(
            # nn.Linear(5800, 3000),
            nn.Linear(Windows_num * self.middle_size * Head_num, 3000),
            nn.ReLU(inplace=True),
            nn.Linear(3000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )

        self.AttentionFFNLn_Kandell_32 = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 500, last_layer=False),
            AttentionWithFFNAndLn(500 * Head_num, 500 * Head_num * ffn_hidden_mult, 500 * Head_num, 500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 500 * Head_num, 500, 200, last_layer=False),
            AttentionWithFFNAndLn(200 * Head_num, 200 * Head_num * ffn_hidden_mult, 200 * Head_num, 200 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 200 * Head_num, 200, self.middle_size, last_layer=False),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )

        self.lstm = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(Head_num, 2)
        # self.PositionalEncoding = PositionalEncoding(1024, 0.1, 116)

    # ffn & layernorm with self-attention

    def attention_with_ffn_and_ln(self, x):
        x = x.float()
        log.info("AttentionFFNLn_Kandell_32 模块输入 x", x)
        # x = self.PositionalEncoding(x)
        x = self.AttentionFFNLn_Kandell_32(x)  # [2, 116, 300]
        log.info("AttentionFFNLn_Kandell_32 模块输出", x)
        x = x.reshape(x.shape[0], -1)
        log.info("展平后维度 x", x)
        output = self.desc_fc(x)
        log.info("[结束] 结果维度 output", output)
        return output
