# ASD_SelfAttention
Judge ASD by self-attention

2022/12/29说明：

    1.新增早停机制，当损失连续10轮高于最低损失时停止当前折训练
    
    2.改进网络层，取消了self-attention层间的线形层，现在降维由V矩阵实现
    
    3.改进结果输出，现在每一折都会输出csv文件以及三个图片：一个acc图象和两个loss图象，训练完成后会输出5折的平均图象

    4.新增了多个模型评价指标：灵敏度、特异性、ROC图形绘制、ROC包围面积AUC

    5.在args中新增了多个可供修改的参数

    6.新增Flood机制（还没试过，默认关闭状态，理解好了再尝试）

    7.新增读取预训练模型（第一次运行会在训练50轮的时候将模型状态保存到pretrain_module/中，随后在args中可以打开读取预训练模型的选项）

    
ToDoList:

    2.在训练前进行PCA降维
    
    3.改进损失函数


代码文件在code，注意改一下主函数code/args.py里面的数据路径和标签路径

所有调整的参数都在code/args.py中

所有需要的包在code/requirements.py中

autodl有专门存数据的地方/root/autodl-tmp/，官方说放在这会更快，同时也方便迁移到别的机器上（确实方便）

标签在description/label_674.csv

直接读入即可，二者顺序已对准

description/label.csv中的reason： 0代表正常，1代表时间点过少舍去，2代表有缺失值舍去

DataProcessing中的代码编写并不严格，不要运行


