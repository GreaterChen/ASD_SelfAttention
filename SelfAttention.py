import math
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        '''
        Self-Attention模块，注释部分为尚未用到，代码参考Zotero上CSDN的代码
        :param num_attention_heads: 多头注意力中的头数，以老师的意思不建议多头，因为参数过多模型欠拟合
        :param input_size: Self-Attention输入层的数量，此处为时间窗的数量
        :param hidden_size:这个好像设置什么都OK，决定了QKV矩阵的尺寸，一般为N*N的就取值为num_attention_heads*input_size
        '''
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        # QKV三个矩阵变换
        self.query_layer = nn.Linear(input_size, self.all_head_size)
        self.key_layer = nn.Linear(input_size, self.all_head_size)
        self.value_layer = nn.Linear(input_size, self.all_head_size)

        # self.dense = nn.Linear(self.attention_head_size, self.attention_head_size)
        # self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        # self.fusion = nn.Linear(num_attention_heads, 1)

    def trans_to_multiple_head(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.to(torch.float32)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        query_heads = self.trans_to_multiple_head(query)
        key_heads = self.trans_to_multiple_head(key)
        value_heads = self.trans_to_multiple_head(value)

        attention_scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        # context = context.permute(0, 1, 3, 2)
        # context = self.fusion(context)
        # context = context.permute(0, 1, 3, 2)
        # context = context.view((1, context.size()[1], 7875))
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)
        # hidden_states = self.dense(context)
        # hidden_states = self.LayerNorm(hidden_states + x)
        return context
