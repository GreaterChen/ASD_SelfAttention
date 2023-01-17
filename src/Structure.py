from SelfAttention import *
from requirements import *
from args import *


class Structure(nn.Module):
    def __init__(self):
        super(Structure, self).__init__()

        self.middle_size = 50
        # 展开、降维、softmax模块
        self.desc_fc = nn.Sequential(
            nn.Linear(Windows_num * self.middle_size * Head_num, 5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )

        self.AttentionFFNLn_Kandell_25 = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 300, last_layer=False),
            AttentionWithFFNAndLn(300 * Head_num, 300 * Head_num * ffn_hidden_mult, 300 * Head_num, 300 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 300 * Head_num, 300, 100, last_layer=False),
            AttentionWithFFNAndLn(100 * Head_num, 100 * Head_num * ffn_hidden_mult, 100 * Head_num, 100 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 100 * Head_num, 100, self.middle_size, last_layer=True),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
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

    # ffn & layernorm with self-attention
    def attention_with_ffn_and_ln(self, x):
        x = x.float()
        x = self.AttentionFFNLn_Kandell_32(x)  # [2, 116, 300]

        x = x.reshape(x.shape[0], -1)
        output = self.desc_fc(x)
        return output
