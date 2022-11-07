from torch import nn

from SelfAttention import SelfAttention

batch_size = 4  # 每次训练样本数
Head_num = 2  # self-attention的头数
Windows_num = 145  # 时间窗的个数
Vector_len = int(116 * 115 / 2)  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)
epoch = 50  # 训练轮次
learn_rate = 0.1


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.middle_size = -1
        # 注意力模块
        self.Attention = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len * Head_num),  # self-attention的输入输出shape一样
            nn.Linear(Vector_len * Head_num, 4000),  # 6670降4000
            nn.ReLU(),
            SelfAttention(Head_num, 4000, 4000 * Head_num),
            nn.Linear(4000 * Head_num, 500),  # 4000降2000
            nn.ReLU(),
            # SelfAttention(Head_num, 2000, 2000 * Head_num),
            # nn.Linear(2000 * Head_num, 500),  # 200降500
            # nn.ReLU(),
            SelfAttention(Head_num, 500, 500 * Head_num),
            nn.Linear(500 * Head_num, 50),  # 500降50
        )

        # 展开、降维、softmax模块
        self.GetRes = nn.Sequential(
            nn.Linear(7250, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.Attention(x)
        x = x.view(x.shape[0], -1)  # 对每个样本展开成向量，7250和注意力模块最后的维度有关系，后续可能需要改一下让他自适应，暂时需要手改
        output = self.GetRes(x)
        return output
