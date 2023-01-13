# 超参数
dataset_size = -1  # 训练的样本总数,-1代表全部训练,调试的时候可以改小点
batch_size = 2  # batch_size
Head_num = 4  # self-attention的头数
epoch = 300  # 最多训练轮次，如果开早停了这个设大点没有影响
learn_rate = 0.0001  # 初始学习率
dropout = 0.5  # 每一个AttentionWithFFNAndLn模块的dropout比例
ffn_hidden_mult = 2  # 隐藏层映射到高维的倍数  隐藏层大小 = 输入层大小 * ffn_hidden_mult

# 其他设置
# root_path = "../raw_data/rois_aal_pkl_pearson"
root_path = "/root/autodl-tmp/rois_aal_pkl_pearson"
label_path = "../description/label_674.csv"

seed = 99335  # 随机数种子，请在良辰吉日先拜三拜后再更改

fisher_r2z = False  # 是否开启Fisher r-to-z 转化

kendall = True
kendall_nums = 32 * 32

pin_memory = False  # 用于dataloader加速训练，但是会增大内存使用量
num_workers = 4  # dataloader的线程数
pre_train = False  # 是否采用预训练模型

EarlyStop = True  # 是否采用早停策略
EarlyStop_patience = 15  # 能容忍多少次测试集损失值无下降
EarlyStop_epoch = 10  # 从多少轮开始启用早停策略（若刚开始就使用可能会导致过早的训练停止）

# 常量
Windows_num = 116  # 时间窗的个数
Vector_len = int(116 * 115 / 2) if not kendall else kendall_nums  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)
