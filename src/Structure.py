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
            nn.Linear(Windows_num * self.middle_size * Head_num , 3000),
            nn.ReLU(inplace=True),
            nn.Linear(3000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )

        self.desc_better_but_slower = nn.Sequential(
            nn.Linear(Windows_num * self.middle_size * Head_num, 10000),
            nn.ReLU(inplace=True),
            nn.Linear(10000, 3000),
            nn.ReLU(inplace=True),
            nn.Linear(3000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )

        self.s1 = SelfAttention(Head_num, Vector_len, Vector_len, 500, last_layer=False)
        self.lstm1 = nn.LSTM(500 * Head_num, 500, batch_first=True, dropout=dropout, num_layers=2)
        self.a1 = AttentionWithFFNAndLn(500 * Head_num, 500 * Head_num * ffn_hidden_mult, 500 * Head_num,
                                        500 * Head_num,
                                        dropout)

        self.s2 = SelfAttention(Head_num, 500, 500, 200, last_layer=False)
        self.lstm2 = nn.LSTM(200 * Head_num, 200, batch_first=True, dropout=dropout, num_layers=2)
        self.a2 = AttentionWithFFNAndLn(200 * Head_num, 200 * Head_num * ffn_hidden_mult, 200 * Head_num,
                                        200 * Head_num,
                                        dropout)

        self.s4 = SelfAttention(Head_num, 200, 200, 100, last_layer=False)
        self.lstm4 = nn.LSTM(100 * Head_num, 100, batch_first=True, dropout=dropout, num_layers=2)
        self.a4 = AttentionWithFFNAndLn(100 * Head_num, 100 * ffn_hidden_mult * Head_num, 100 * Head_num,
                                        100 * Head_num, dropout)

        self.s3 = SelfAttention(Head_num, 200, 200, self.middle_size, last_layer=True)
        self.lstm3 = nn.LSTM(self.middle_size * Head_num, self.middle_size, batch_first=True, dropout=dropout,
                             num_layers=2)
        self.a3 = AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * ffn_hidden_mult * Head_num,
                                        self.middle_size * Head_num, self.middle_size * Head_num, dropout)

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

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(Head_num, 2)

    # ffn & layernorm with self-attention

    def lstm_attention(self, x):
        x = self.s1(x)
        x = self.a1(x)
        x, (_, _) = self.lstm1(x)
        x = self.s2(x)
        x = self.a2(x)
        x, (_, _) = self.lstm2(x)
        x = self.s3(x)
        x = self.a3(x)
        x, (_, _) = self.lstm3(x)
        return x

    def attention_with_ffn_and_ln(self, x):
        x = x.float()
        x = self.AttentionFFNLn_Kandell_32(x)  # [2, 116, 300]
        x = x.reshape(x.shape[0], -1)
        output = self.desc_fc(x)
        return output
