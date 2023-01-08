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
        self.FC0 = nn.Sequential(
            nn.Linear(Windows_num * self.middle_size * Head_num, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )

        # 对二维卷积的尝试
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 2, (5, 5), stride=(1, 1)),  # (2,46,112)
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),  # (2,23,56)
            nn.Conv2d(2, 4, (5, 5), stride=(1, 1)),  # (4,19,52)
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),  # (4,9,26)
        )

        self.desc_conv2 = nn.Sequential(
            nn.Linear(234 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.Softmax(dim=1)
        )

        # 对一维卷积的尝试
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=116, out_channels=64, kernel_size=tuple([5])),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=tuple([5])),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=tuple([5])),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=tuple([5])),
        )

        self.desc_conv1 = nn.Sequential(
            nn.Linear(272, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Softmax(dim=1)

        )

        self.AttentionFFNLn = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 4000),  # self-attention的输入输出shape一样
            AttentionWithFFNAndLn(4000, 4000*2, 4000, 4000, 0.9),
            SelfAttention(Head_num, 4000 * Head_num, 4000, 500),
            AttentionWithFFNAndLn(500, 500 * 2, 500, 500, 0.9),
            SelfAttention(Head_num, 500 * Head_num, 500, self.middle_size),
            AttentionWithFFNAndLn(self.middle_size, self.middle_size * 2, self.middle_size, self.middle_size, 0.9),
        )

    # 二维卷积降维
    def Conv2(self, x):
        x = self.Attention(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, 50, 116)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        output = self.desc_conv2(x)
        return output

    # 全连接降维
    def FC(self, x):
        x = self.Attention(x)
        x = x.view(x.shape[0], -1)
        output = self.FC0(x)
        return output

    # 一维卷积降维
    def Conv1(self, x):
        x = self.Attention(x)
        x = self.conv1(x)  # (2,8,34)
        x = x.view(x.shape[0], -1)
        output = self.desc_conv1(x)
        return output

    def attention_with_ffn_and_ln(self, x):
        x = self.AttentionFFNLn(x)
        x = x.view(x.shape[0], -1)
        output = self.FC0(x)
        return output
