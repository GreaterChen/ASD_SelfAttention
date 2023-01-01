import torch.optim.lr_scheduler

from requirements import *
from args import *
from utils import draw_result_pic, EarlyStopping, GetAvg, GetData, Draw_ROC
from Module import Module

from torch.cuda.amp import autocast, GradScaler


def Train():
    global Y_train, Y_pred, epoch_i, auc_list
    all_data = GetData(root_path, label_path, dataset_size)  # 一次性读取所有数据

    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 初始化5折交叉验证的工具

    # 对每一折进行记录
    train_acc_list_kf = []
    train_loss_list_kf = []
    test_acc_list_kf = []
    test_loss_list_kf = []

    SEN_list_kf = []
    SPE_list_kf = []
    auc_pd = pd.DataFrame()

    split_range = 0
    last_time = time.time()
    k = 0  # 表征第几折
    for train_index, test_index in kf.split(all_data):  # 此处获取每一折的索引
        # 对于每一折来说，都要从0开始训练模型
        # 因为如果不同折训练同一个模型，会出现当前折的测试集曾被另一折当作训练集训练，导致准确率异常
        if pre_train:
            module = torch.load(f"../pretrain_module/pretrain_{k}.pt")
        else:
            module = Module()
        module = module.cuda()

        # 损失函数：交叉熵
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.cuda()
        # 优化器：SGD
        optimizer = torch.optim.SGD(module.parameters(), lr=learn_rate)

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.5, total_steps=100)

        scaler = GradScaler()

        early_stop = EarlyStopping(patience=EarlyStop_patience)

        p_table = PrettyTable(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "SEN", "SPE", "lr","time(s)"])

        # 此处获取真正的该折数据
        train_fold = Subset(all_data, train_index)
        test_fold = Subset(all_data, test_index)

        train_size = len(train_index)
        test_size = len(test_index)

        train_dataloader = DataLoader(train_fold, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=True)
        test_dataloader = DataLoader(test_fold, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     pin_memory=True)

        split_range += 1

        # 对该折的所有轮进行记录
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []

        SEN_list = []
        SPE_list = []
        auc_list = []

        # 下面开始当前折的训练
        for epoch_i in range(epoch):
            if early_stop.early_stop:
                break
            # 对该折该轮的所有dataloader进行记录
            epoch_train_loss = 0
            epoch_train_acc = 0
            epoch_test_loss = 0
            epoch_test_acc = 0

            # 评价指标
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            Y_pred = []
            Y_train = []

            module.train()
            # 下面开始当前折、当前轮的训练，即以batch_size的大小进行训练
            for data in tqdm(train_dataloader, desc=f'train-Fold{split_range}-Epoch{epoch_i + 1}', file=sys.stdout):
                if early_stop.early_stop:
                    print("触发早停")
                    break
                x, y = data
                x = x.cuda()
                y = y.cuda()
                y = y.to(torch.float32)  # 这一步似乎很费时间
                with autocast():
                    output = module(x)
                    loss = loss_fn(output, y)
                epoch_train_loss += loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # scaler.step(scheduler)
                scaler.step(optimizer)
                # optimizer.step()
                scaler.update()
                train_acc = 0
                for i, res in enumerate(output):
                    if res[0] > res[1]:
                        train_acc = train_acc + 1 if y[i][0] > y[i][1] else train_acc
                    if res[0] < res[1]:
                        train_acc = train_acc + 1 if y[i][0] < y[i][1] else train_acc
                epoch_train_acc += train_acc

            # 记录当前折、当前轮的数据
            train_acc_list.append(float(epoch_train_acc / train_size))
            train_loss_list.append(float(epoch_train_loss))

            module.eval()  # 设置测试模式
            with torch.no_grad():
                for data in tqdm(test_dataloader, desc=f'test-Fold{split_range}-Epoch{epoch_i + 1}', file=sys.stdout):
                    x, y = data
                    x = x.cuda()
                    y = y.cuda()
                    y = y.to(torch.float32)

                    output = module(x)

                    loss = loss_fn(output, y)
                    epoch_test_loss += loss
                    test_acc = 0
                    for i, res in enumerate(output):
                        Y_pred.append(res[0])
                        Y_train.append(y[i][0])
                        if res[0] > res[1]:
                            if y[i][0] > y[i][1]:  # 预测有病实际有病
                                TP += 1
                            else:  # 预测有病实际没病
                                FP += 1
                        if res[0] < res[1]:
                            if y[i][0] > y[i][1]:  # 预测没病实际有病
                                FN += 1
                            else:  # 预测没病实际没病
                                TN += 1
                    epoch_test_acc += test_acc

            ACC = (TP + TN) / (TP + TN + FP + FN)  # 准确率
            SEN = TP / (TP + FN)  # 灵敏度
            SPE = TN / (TN + FP)  # 特异性

            SEN_list.append(SEN)
            SPE_list.append(SPE)
            test_acc_list.append(ACC)
            test_loss_list.append(float(epoch_test_loss))

            if (epoch_i + 1) % 10 == 0:
                auc = Draw_ROC(Y_train, Y_pred, epoch_i + 1, k + 1)
                auc_list.append(auc)

            if epoch_i + 1 == 50:
                torch.save(module.state_dict(), f"../pretrain_module/pretrain_{k + 1}.pt")

            p_table.add_row(
                [epoch_i + 1,
                 format(float(train_loss_list[-1]), '.3f'),
                 format(float(train_acc_list[-1]), '.3f'),
                 format(float(test_loss_list[-1]), '.3f'),
                 format(float(test_acc_list[-1]), '.3f'),
                 format(float(SEN), '.3f'),
                 format(float(SPE), '.3f'),
                 format(float(scheduler.get_lr()[0]),'.3f'),
                 format(float(time.time() - last_time), '2f')])

            last_time = time.time()
            print(p_table)
            if EarlyStop:
                if epoch_i >= 50:
                    early_stop(epoch_test_loss, module)

        auc = Draw_ROC(Y_train, Y_pred, epoch_i + 1, k + 1)
        auc_list.append(auc)
        auc_pd = pd.concat([pd.DataFrame({f'第{k + 1}轮': auc_list}), auc_pd])

        draw_result_pic([train_acc_list, test_acc_list], 0, f'{k + 1}Fold_acc')
        draw_result_pic([train_loss_list, test_loss_list], 0, f'{k + 1}Fold_loss')

        # 记录每一折的数据，是一个二维的列表
        train_acc_list_kf.append(train_acc_list)
        train_loss_list_kf.append(train_loss_list)
        test_acc_list_kf.append(test_acc_list)
        test_loss_list_kf.append(test_loss_list)

        SEN_list_kf.append(SEN_list)
        SPE_list_kf.append(SPE_list)

        K_Fold_res = pd.DataFrame()
        K_Fold_res['训练集损失值'] = train_loss_list_kf[k]
        K_Fold_res['训练集准确率'] = train_acc_list_kf[k]
        K_Fold_res['测试集损失值'] = test_loss_list_kf[k]
        K_Fold_res['测试集准确率'] = test_acc_list_kf[k]
        K_Fold_res['灵敏度'] = SEN_list_kf[k]
        K_Fold_res['特异性'] = SPE_list_kf[k]
        K_Fold_res.to_csv(f"../result/{k + 1}_Fold.csv")
        k += 1

    avg_train_acc = GetAvg(train_acc_list_kf)
    avg_train_loss = GetAvg(train_loss_list_kf)
    avg_test_acc = GetAvg(test_acc_list_kf)
    avg_test_loss = GetAvg(test_loss_list_kf)

    avg_sen = GetAvg(SEN_list_kf)
    avg_spe = GetAvg(SPE_list_kf)

    auc_pd.to_csv("../res/auc.scv")

    res = pd.DataFrame()
    res['训练集准确率'] = avg_train_acc
    res['测试集准确率'] = avg_test_acc
    res['训练集损失值'] = avg_train_loss
    res['测试集损失值'] = avg_test_loss
    res['灵敏度'] = avg_sen
    res['特异性'] = avg_spe
    res.to_csv("../result/result.csv")

    # 传结果list格式： [train(list), test(list)]
    draw_result_pic(res=[avg_train_acc, avg_test_acc],
                    start_epoch=0,
                    pic_title='acc')
    draw_result_pic(res=[avg_train_loss, avg_test_loss],
                    start_epoch=0,
                    pic_title='loss')


if __name__ == '__main__':
    if not os.path.exists("../result"):
        os.makedirs("../result")
    if not os.path.exists("../pretrain_module"):
        os.makedirs("../pretrain_module")

    start_time = time.time()
    Train()
    end_time = time.time()
    spend_time = end_time - start_time
    min = spend_time // 60
    sec = int(spend_time - 60 * min)
    print(f"模型训练总用时:{min}分{sec}秒")
