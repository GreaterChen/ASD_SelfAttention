# ASD_SelfAttention
Judge ASD by self-attention

### 重要说明！！！！！！！

    1.连到服务器跑的时候可能会报错说不能在cuda上跑，因此需要改源文件配置：
    
        在服务器中： vim /root/miniconda3/lib/python3.8/site-packages/torch/_tensor.py

        然后在vim中的命令行模式进行查找： /self.numpy()

        然后进入编辑模式改为： self.cpu().numpy()

        然后:wq退出即可

    2.由于之前忽略了read_csv()会把第一行数据默认当作标签,导致时间窗少了一个，请在服务器中运行且仅运行一遍code/fix_pkl.py
        -> 注意文件中的root_path变量指示的位置是否正确

    3. 运行前注意修改args.py中的root_path,learning_rate等参数

### 项目说明

    1.代码文件在src，所有调整的参数都在src/args.py中

    3.autodl有专门存数据的地方/root/autodl-tmp/，官方说放在这会更快，同时也方便迁移到别的机器上（确实方便）
    
    4.标签在description/label_674.csv
    
    5.description/label.csv中的reason： 0代表正常，1代表时间点过少舍去，2代表有缺失值舍去

### 更新说明
#### 2020/1/17
    1.新增正负预测率评价指标，简化prettytable展示指标
    
    2.新增训练从指定折到指定折，在args中的begin_flod 和 end_flod进行控制

    3.修复softmax重复的bug

#### 2023/1/14
    1.新增文件Regularization,可以对模型施加L1和L2正则化，在args里面可以选择,初步尝试L1设置为1e-7，L2设置为1e-4

    2.恢复了混合精度训练，现在3090 24G不会爆显存了，2080ti可以训一折

    3.今日总结：
        (1) 自编码特征提取效果较差、LSTM没有明显效果
        (2) batch_size = 2,head_num = 6,lr = 1e-4为最优参数
        (3) Fisher r_to_z 能够提高模型稳定性，对平均准确率有较大提升，建议开启
#### 2023/1/13
    1.今天发现之前attention层后的降维直接从两万多降到了500，今天进行了优化，增大了参数量

    2.模型默认包含LSTM结构，如果想删去，在Structure.py中attention_with_ffn_and_ln()中进行更改

    3.在每一折开始时添加torch.cuda.empty_cache()，用于清空显存缓存，防止第二折爆显存

    4.args中增加Fisher's r-to-z transform，直观感受是较小的损失下降能带来较大的准确率上升，但是训到最后训练集都不下降了，上限略低

    5.对SGD新增动量，参数设置为0.9时收敛过快，在lr=1e-4时损失波动非常大，lr=1e-5时和不加区别不大

    6.尝试CapsNet，玩不明白，效果很差

#### 2023/1/12
    1.args新增对kendall降维后特征数量的设置，目前只支持32*32、56*56、667，若想尝试其他数字可以在Structure中仿照已有结构更改

    2.若损失值连续5轮不下降，学习率会乘上0.2

    3.移除混合精度训练，由于模型训练速度已经很快了，混合精度并没有带来提升，为了防止隐藏的问题，故移除

    4.在self_attention层前添加LSTM层，现有设置是Kendall从6670降到1024，LSTM从1024降到512，然后再接self_attention

    5.尝试了使用LSTM接到self_attention层后面代替全连接，波动非常大。

#### 2023/1/11:
    1.新加两个文件完成了Kendall秩相关系数，未参与模型训练，后续移除
        -> Kendall_prepare.py 准备pearson文件
        -> Kendall.py   进行运算
    暂时使用全部的时间序列（不考虑时间窗）进行特征提取，后续打算对每个时间窗都进行计算，取综合最优特征

    2.移除了Structure.py中无用模块，新增AttentionFFNLn_Kandell模块
#### 2023/1/8:
    1.SelfAttention.py下新增三个类，详细说明见代码注释
        -> FFN  
        -> AddNorm
        -> AttentionWithFFNAndLn    连接FFN和AddNorm，组合成一个模块，供调用
    2.Structure.py下新增
        -> 带有AttentionWithFFNAndLn模块的self.AttentionFFNLn网络
        -> forward函数 def attention_with_ffn_and_ln(self, x)
#### 2023/1/3：

    1.尝试不展平使用二维卷积和一维卷积，loss下降一点后就不变了（梯度消失？）

    2.解决了不同batch_size下loss不一致的问题：交叉熵计算出的loss是平均值，因此需要乘上batch_size才是实际损失
#### 2023/1/2：

    1.在数据读取中将torch.Tensor()改为torch.to_tensor(),读取速度有明显提升，现在5秒就能读取完全部数据

    2.尝试MSE损失函数，效果不如交叉熵

    3.尝试LeakyReLU,和ReLU区别不大

#### 2023/1/1：

    1.新增混合精度训练，进一步提升了模型训练速度

    2.暂时删去Flood策略

    3.主函数中保留了OneCycleLR的代码，但似乎无法和混合精度训练同时使用，故未使用

#### 2022/12/29：

    1.新增早停机制，当损失连续10轮高于最低损失时停止当前折训练
    
    2.改进网络层，取消了self-attention层间的线形层，现在降维由V矩阵实现
    
    3.改进结果输出，现在每一折都会输出csv文件以及三个图片：一个acc图象和两个loss图象，训练完成后会输出5折的平均图象

    4.新增了多个模型评价指标：灵敏度、特异性、ROC图形绘制、ROC包围面积AUC

    5.在args中新增了多个可供修改的参数

    6.新增Flood机制（还没试过，默认关闭状态，理解好了再尝试）

    7.新增读取预训练模型（第一次运行会在训练50轮的时候将模型状态保存到pretrain_module/中，随后在args中可以打开读取预训练模型的选项）



