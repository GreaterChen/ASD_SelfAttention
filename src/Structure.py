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

        self.conv3d1 = nn.Conv3d(1, 1, kernel_size=(Head_num, 5, 1), stride=(1, 1, 1))
        self.batchnorm1 = nn.BatchNorm3d(1)

        self.AttentionFFNLn_Kandell_56 = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 1500, last_layer=False),
            AttentionWithFFNAndLn(1500 * Head_num, 1500 * Head_num * ffn_hidden_mult, 1500 * Head_num, 1500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 1500 * Head_num, 1500, 500, last_layer=False),
            AttentionWithFFNAndLn(500 * Head_num, 500 * Head_num * ffn_hidden_mult, 500 * Head_num, 500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 500 * Head_num, 500, self.middle_size, last_layer=True),
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

            SelfAttention(Head_num, 200 * Head_num, 200, self.middle_size, last_layer=True),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )
        self.lstm = nn.LSTM(6, 6, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # ffn & layernorm with self-attention
    def attention_with_ffn_and_ln(self, x):
        x = x.float()
        x = self.AttentionFFNLn_Kandell_32(x)  # [2, 116, 300]

        x = x.reshape(x.shape[0], -1)
        output = self.desc_fc(x)
        return output
