from SelfAttention import *
from requirements import *
from args import *


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
            nn.Linear(Windows_num * self.middle_size * Head_num, 5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )

        self.desc_2 = nn.Sequential(
            nn.Linear(6960, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )

        self.AttentionFFNLn_Kandell_56 = nn.Sequential(
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

        self.AttentionFFNLn_Kandell_32 = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 500),
            AttentionWithFFNAndLn(500 * Head_num, 500 * Head_num * ffn_hidden_mult, 500 * Head_num, 500 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 500 * Head_num, 500, 200),
            AttentionWithFFNAndLn(200 * Head_num, 200 * Head_num * ffn_hidden_mult, 200 * Head_num, 200 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 200 * Head_num, 200, self.middle_size),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )

        self.AttentionFFNLn_Kandell_512 = nn.Sequential(
            SelfAttention(Head_num, 512, 512, 256),
            AttentionWithFFNAndLn(256 * Head_num, 256 * Head_num * ffn_hidden_mult, 256 * Head_num, 256 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 256 * Head_num, 256, 100),
            AttentionWithFFNAndLn(100 * Head_num, 100 * Head_num * ffn_hidden_mult, 100 * Head_num, 100 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 100 * Head_num, 100, self.middle_size),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )

        self.lstm = nn.LSTM(300, 300, batch_first=True)

    # 全连接降维
    def FC(self, x):
        x = self.Attention(x)
        x = x.view(x.shape[0], -1)
        output = self.desc_fc(x)
        return output

    # ffn & layernorm with self-attention
    def attention_with_ffn_and_ln(self, x):
        x = x.float()
        # x, (_, _) = self.lstm(x)

        # sae = torch.load("SAE.pth")
        # sae.eval()
        # sae.cuda()
        # x = sae.encoder(x)
        x = self.AttentionFFNLn_Kandell_32(x)
        # x, (_, _) = self.lstm(x)
        # x = x.view(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        output = self.desc_fc(x)
        return output
