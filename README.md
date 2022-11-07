# ASD_SelfAttention
Judge ASD by self-attention

原始数据在raw_data/rois_aal_csv

Pearson数据过大，需先运行pearson_calculate.py

结果在raw_data/rois_aal_csv_pearson

标签在description/label.csv

直接读入即可，二者顺序已对准

label中的reason： 0代表正常，1代表时间点过少舍去，2代表有缺失值舍去

DataProcessing中的代码编写并不严格，不要运行


