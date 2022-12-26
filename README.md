# ASD_SelfAttention
Judge ASD by self-attention

代码文件在code，注意改一下主函数code/main*.py里面的数据路径和标签路径

原始数据在raw_data/rois_aal_csv

Pearson数据过大，需先运行pearson_calculate.py（或者找我要压缩包文件然后直接上传再解压）

结果在raw_data/rois_aal_csv_pearson（似乎autodl有专门存数据的地方/root/autodl-tmp/，放在这可能会更快？没试过）

标签在description/label_678.csv

直接读入即可，二者顺序已对准(可以运行一下code/check_order.py来确定对准了)

description/label.csv中的reason： 0代表正常，1代表时间点过少舍去，2代表有缺失值舍去

DataProcessing中的代码编写并不严格，不要运行


