# ASD_SelfAttention
Judge ASD by self-attention

代码文件在code，注意改一下主函数code/args.py里面的数据路径和标签路径

所有调整的参数都在code/args.py中

所有需要的包在code/requirements.py中

autodl有专门存数据的地方/root/autodl-tmp/，官方说放在这会更快，同时也方便迁移到别的机器上（确实方便）

标签在description/label_674.csv

直接读入即可，二者顺序已对准

description/label.csv中的reason： 0代表正常，1代表时间点过少舍去，2代表有缺失值舍去

DataProcessing中的代码编写并不严格，不要运行


