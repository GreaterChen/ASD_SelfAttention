dataset_size = -1  # 训练的样本总数,-1代表全部训练,调试的时候可以改小点
batch_size = 2  # batch_size
Head_num = 6  # self-attention的头数
epoch = 100  # 最多训练轮次，如果开早停了这个设大点没有影响
learn_rate = 1e-3  # 初始学习率
dropout = 0.5  # 每一个AttentionWithFFNAndLnn模块的dropout比例
ffn_hidden_mult = 5  # 隐藏层映射到高维的倍数  隐藏层大小 = 输入层大小 * ffn_hidden_mult
k_fold = 5
institutions = ['KKI', 'LEUVEN', 'NYU', 'STANFORD', 'Trinity', 'UM', 'USM', 'Yale']  # 做留一站点验证验证的机构
static = True

L1_en = False
L1_weight_decay = 1e-3  # L1正则化参数
L2_en = False
L2_weight_decay = 0.005  # L2正则化参数

decay = 0.8  # 每次衰减decay倍学习率

begin_fold = 1  # 开始训练的轮数
end_fold = 10  # 结束训练的轮数

seed = 99335  # 随机数种子，请在良辰吉日先拜三拜后再更改 99335

fisher_r2z = True  # 是否开启Fisher r-to-z 转化

kendall = True
kendall_nums = 1024

pin_memory = True  # 用于dataloader加速训练，但是会增大内存使用量
num_workers = 4  # dataloader的线程数
pre_train = False  # 是否采用预训练模型

EarlyStop = True  # 是否采用早停策略
EarlyStop_patience = 30  # 能容忍多少次测试集损失值无下降
EarlyStop_epoch = 10  # 从多少轮开始启用早停策略（若刚开始就使用可能会导致过早的训练停止）

# 常量
Windows_num = 116  # 时间窗的个数
Vector_len = int(116 * 115 / 2) if not kendall else kendall_nums  # 上三角展开后的长度
# 306 368

# root_path = "../../raw_data/rois_aal_pkl_pearson"
# root_path = "/root/autodl-tmp/rois_aal_pkl_pearson"
root_path = "/root/autodl-tmp/rois_aal_pearson_static_pkl"
label_path = "../description/label_674.csv"

# 日志显示级别 供调试使用
LOGGER_DISPLAY = False  # True显示日志, False不显示
LOGGER_DISPLAY_VAR = False  # True显示[变量+变量形状], False显示[变量形状]
