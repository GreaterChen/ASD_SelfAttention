from requirements import *
from SelfAttention import SelfAttention
from args import *


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.middle_size = 50
        # 注意力模块
        self.Attention = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len, 4000),  # self-attention的输入输出shape一样
            SelfAttention(Head_num, 4000 * Head_num, 4000, 500),
            SelfAttention(Head_num, 500 * Head_num, 500, self.middle_size),
        )

        # 展开、降维、softmax模块
        self.GetRes = nn.Sequential(
            nn.Linear(Windows_num * self.middle_size * Head_num, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.Attention(x)
        x = x.view(x.shape[0], -1)
        output = self.GetRes(x)
        return output
