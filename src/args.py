# 超参数
dataset_size = -1  # 训练的样本总数,-1代表全部训练,调试的时候可以改小点
batch_size = 2  # batch_size
Head_num = 2  # self-attention的头数
epoch = 200  # 最多训练轮次，极大概率会早停，这个设大点也ok
learn_rate = 0.0001  # 学习率
dropout = 0.9   # 每一个AttentionWithFFNAndLn模块的dropout比例
ffn_hidden_mult = 2     # 隐藏层映射到高维的倍数  隐藏层大小 = 输入层大小 * ffn_hidden_mult

# 其他设置
# root_path = "../raw_data/rois_aal_pkl_pearson"
root_path = "/root/autodl-tmp/rois_aal_pkl_pearson"
label_path = "label_674.csv"

kendall = True

pin_memory = True   # 用于dataloader加速训练，但是会增大内存使用量
num_workers = 8    # dataloader的线程数
pre_train = False    # 是否采用预训练模型

EarlyStop = True    # 是否采用早停策略
EarlyStop_patience = 10  # 能容忍多少次测试集损失值无下降
EarlyStop_epoch = 30    # 从多少轮开始启用早停策略（若刚开始就使用可能会导致过早的训练停止）

# 常量
Windows_num = 116  # 时间窗的个数
Vector_len = int(116 * 115 / 2) if not kendall else 3136  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)





