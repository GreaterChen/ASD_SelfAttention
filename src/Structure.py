from SelfAttention import *
from requirements import *
from args import *


# 包含尝试过的架构：全连接降维、一维卷积降维、二维卷积降维
class Structure(nn.Module):
    def __init__(self):
        super(Structure, self).__init__()

        self.middle_size = 50
        # 注意力模块
        self.Attention = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 4000),  # self-attention的输入输出shape一样
            SelfAttention(Head_num, 4000 * Head_num, 4000, 500),
            SelfAttention(Head_num, 500 * Head_num, 500, self.middle_size),
        )

        # 展开、降维、softmax模块
        self.desc_fc = nn.Sequential(
            nn.Linear(Windows_num * self.middle_size * Head_num, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )

        self.AttentionFFNLn = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 4000),
            AttentionWithFFNAndLn(4000 * Head_num, 4000 * Head_num * ffn_hidden_mult, 4000 * Head_num, 4000 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 4000 * Head_num, 4000, 500),
            AttentionWithFFNAndLn(500 * Head_num, 500 * Head_num * ffn_hidden_mult, 500 * Head_num, 500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 500 * Head_num, 500, self.middle_size),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )

        self.AttentionFFNLn_Kandell = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 1500),
            AttentionWithFFNAndLn(1500 * Head_num, 1500 * Head_num * ffn_hidden_mult, 1500 * Head_num, 1500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 1500 * Head_num, 1500, 500),
            AttentionWithFFNAndLn(500 * Head_num, 500 * Head_num * ffn_hidden_mult, 500 * Head_num, 500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 500 * Head_num, 500, self.middle_size),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )

    # 全连接降维
    def FC(self, x):
        x = self.Attention(x)
        x = x.view(x.shape[0], -1)
        output = self.desc_fc(x)
        return output

    # ffn & layernorm with self-attention
    def attention_with_ffn_and_ln(self, x):
        x = self.AttentionFFNLn_Kandell(x)
        x = x.view(x.shape[0], -1)
        output = self.desc_fc(x)
        return output
