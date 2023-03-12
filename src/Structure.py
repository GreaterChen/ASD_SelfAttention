import logging

from SelfAttention import *
from requirements import *
from args import *
from utils import *


class StackAutoEncoder(nn.Module):
    def __init__(self, encoders):
        super(StackAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(encoders[0], encoders[1])
        self.decoder = nn.Sequential(encoders[2], encoders[3])

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


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

        self.desc = nn.Sequential(
            # nn.Linear(5800, 3000),
            nn.Linear(116 * 32, 1000),
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

        self.AttentionFFNLn_Kandell_100 = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 50, last_layer=False),
            AttentionWithFFNAndLn(50 * Head_num, 50 * Head_num * ffn_hidden_mult, 50 * Head_num, 50 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 50 * Head_num, 50, 50, last_layer=False),
            AttentionWithFFNAndLn(50 * Head_num, 50 * Head_num * ffn_hidden_mult, 50 * Head_num, 50 * Head_num,
                                  dropout),
            SelfAttention(Head_num, 50 * Head_num, 50, self.middle_size, last_layer=False),
            AttentionWithFFNAndLn(self.middle_size * Head_num, self.middle_size * Head_num * ffn_hidden_mult,
                                  self.middle_size * Head_num, self.middle_size * Head_num, dropout),
        )

        self.lstm = nn.LSTM(1024, 32, batch_first=True, dropout=0.5, num_layers=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(1024, 100)
        self.linear_2 = nn.Linear(100, 10)
        self.linear_3 = nn.Linear(32, 2)
        self.PositionalEncoding = PositionalEncoding(1024, 0.1, 116)

        self.cnn = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.cnn_desc = nn.Sequential(
            nn.Linear(576, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.conv21 = nn.Conv3d(1, 1, 2, 2)

    # ffn & layernorm with self-attention

    def attention_with_ffn_and_ln(self, x):
        x = x.float()
        log.info("AttentionFFNLn_Kandell_32 模块输入 x", x)
        # x = self.PositionalEncoding(x)
        x, (_, _) = self.lstm(x)
        x = self.AttentionFFNLn_Kandell_32(x)  # [2, 116, 300]
        log.info("AttentionFFNLn_Kandell_32 模块输出", x)
        x = x.reshape(x.shape[0], -1)
        log.info("展平后维度 x", x)
        output = self.desc_fc(x)
        log.info("[结束] 结果维度 output", output)
        return output

    def SAE(self, x):
        x = x.float()
        sae = torch.load("SAE.pth")
        sae.eval()
        sae.cuda()
        x = sae.encoder(x)
        x = x.reshape(x.shape[0], -1)
        output = self.linear(x)
        return output

    def LSTM(self, x):
        x = x.float()
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_3(x)
        # x = self.desc(x)
        return x

    def CNN(self, x):
        x = x.float()
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.cnn_desc(x)
        return x