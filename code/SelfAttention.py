from requirements import *


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, dim_qk, dim_v):
        """
        Self-Attention模块
        :param num_attention_heads: 多头注意力中的头数，以老师的意思不建议多头，因为参数过多模型欠拟合
        :param input_size: Self-Attention输入层的尺寸
        :param dim_qk: QK矩阵的维度,不考虑多头
        :param dim_v: V矩阵的维度,不考虑多头(决定了降维的输出尺寸)
        """
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads  # 头数
        self.signle_head_size = input_size  # 单头的尺寸
        self.all_head_size = input_size * num_attention_heads  # 总共的尺寸
        self.norm_fact = 1 / sqrt(dim_qk // num_attention_heads)  # 根号下dk
        self.dim_qk = dim_qk
        self.dim_v = dim_v

        # QKV三个矩阵变换
        self.query_layer = nn.Linear(self.signle_head_size, self.dim_qk * self.num_attention_heads, bias=False)
        self.key_layer = nn.Linear(self.signle_head_size, self.dim_qk * self.num_attention_heads, bias=False)
        self.value_layer = nn.Linear(self.signle_head_size, self.dim_v * self.num_attention_heads, bias=False)

    def trans_to_multiple_head(self, x, sign):
        if sign == 'q' or sign == 'k':
            new_size = x.size()[:-1] + (self.num_attention_heads, self.dim_qk)
        else:  # sign == 'v'
            new_size = x.size()[:-1] + (self.num_attention_heads, self.dim_v)
        x = x.view(*new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.to(torch.float32)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        query_heads = self.trans_to_multiple_head(query, 'q')
        key_heads = self.trans_to_multiple_head(key, 'k')
        value_heads = self.trans_to_multiple_head(value, 'v')

        attention_scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        attention_scores = attention_scores / torch.tensor(sqrt(self.dim_qk // self.num_attention_heads))

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[:-2] + (self.dim_v * self.num_attention_heads,)
        context = context.view(*new_size)
        return context
