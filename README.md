# ASD_SelfAttention
Judge ASD by self-attention

### 重要说明！！！！！！！

    1.连到服务器跑的时候可能会报错说不能在cuda上跑，因此需要改源文件配置：
    
        在服务器中： vim /root/miniconda3/lib/python3.8/site-packages/torch/_tensor.py

        然后在vim中的命令行模式进行查找： /self.numpy()

        然后进入编辑模式改为： self.cpu().numpy()

        然后:wq退出即可

    2.由于之前忽略了read_csv()会把第一行数据默认当作标签,导致时间窗少了一个，请在服务器中运行且仅运行一遍code/fix_pkl.py 

### 项目说明

    1.代码文件在code，注意改一下主函数code/args.py里面的数据路径和标签路径
    
    2.所有调整的参数都在code/args.py中
    
    3.autodl有专门存数据的地方/root/autodl-tmp/，官方说放在这会更快，同时也方便迁移到别的机器上（确实方便）
    
    4.标签在description/label_674.csv
    
    5.description/label.csv中的reason： 0代表正常，1代表时间点过少舍去，2代表有缺失值舍去

### 更新说明
#### 2023/1/3：

    1.尝试不展平使用二维卷积和一维卷积，loss下降一点后就不变了（梯度消失？）
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



