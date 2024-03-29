import torch

from requirements import *
from args import *
from utils import *
from Module import Module
from Regularization import *


def Train():
    global Y_train, Y_pred, epoch_i, auc_list, best_acc, best_acc_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_data = GetData(cc200_path, cc200_label, dataset_size)

    for institution in institutions:
        test_index = []
        train_index = []
        for i in range(len(all_data.label_info)):
            if institution.upper() in all_data.label_info.iloc[i]['SITE_ID']:
                test_index.append(i)
            else:
                train_index.append(i)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)

        # 对每一折进行记录
        train_acc_list_kf = []
        train_loss_list_kf = []
        test_acc_list_kf = []
        test_loss_list_kf = []

        SEN_list_kf = []
        SPE_list_kf = []
        PPV_list_kf = []
        NPV_list_kf = []
        F1_list_kf = []
        AUC_list_kf = []
        FPR_list_kf = []
        TPR_list_kf = []

        best_acc_list = []
        split_range = 0
        last_time = time.time()
        k = 0  # 表征第几折
        for i in range(1):  # 此处获取每一折的索引
            # 对于每一折来说，都要从0开始训练模型
            # 因为如果不同折训练同一个模型，会出现当前折的测试集曾被另一折当作训练集训练，导致准确率异常
            if pre_train:
                module = torch.load(f"../pretrain_module/pretrain_{k}.pt")
            else:
                module = Module()

            module = module.to(device)
            scaler = GradScaler()

            # 损失函数：交叉熵
            loss_fn = nn.CrossEntropyLoss().to(device)

            # 优化器：SGD
            lr = learn_rate
            if L2_en:
                optimizer = torch.optim.SGD(module.parameters(), lr=lr, weight_decay=L2_weight_decay)
            else:
                optimizer = torch.optim.SGD(module.parameters(), lr=lr)

            early_stop = EarlyStopping(patience=EarlyStop_patience)
            roc = ROC()

            p_table = PrettyTable(
                ["institution", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "best_acc", "lr(1e-4)",
                 "AUC", "time(s)"])

            # 此处获取真正的该折数据
            train_fold = Subset(all_data, train_index)
            test_fold = Subset(all_data, test_index)

            train_size = len(train_index)
            test_size = len(test_index)
            print("训练集大小:", train_size)
            print("测试集大小:", test_size)

            train_dataloader = DataLoader(train_fold, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                          pin_memory=pin_memory, drop_last=True)
            test_dataloader = DataLoader(test_fold, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                         pin_memory=pin_memory, drop_last=True)

            split_range += 1

            # 对该折的所有轮进行记录
            train_acc_list = []
            train_loss_list = []
            test_acc_list = []
            test_loss_list = []

            SEN_list = []
            SPE_list = []
            AUC_list = []
            FPR_list = []
            TPR_list = []
            PPV_list = []
            NPV_list = []
            F1_list = []

            best_acc = 0

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
                for data in tqdm(train_dataloader, desc=f'train-Fold{k + 1}-Epoch{epoch_i + 1}', file=sys.stdout):
                    if early_stop.early_stop:
                        print("触发早停")
                        break
                    x, y = data
                    x = x.cuda()
                    y = y.cuda()
                    y = y.to(torch.float32)  # 这一步似乎很费时间

                    with autocast():
                        log.separator()
                        log.info("[开始] 模型输入 x", x)
                        output = module(x)
                        # loss, pro_result = loss_fn(output, y)
                        loss = loss_fn(output, y)
                    epoch_train_loss += loss.item() * batch_size
                    optimizer.zero_grad()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()
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
                    for data in tqdm(test_dataloader, desc=f'test-Fold{k + 1}-Epoch{epoch_i + 1}', file=sys.stdout):
                        x, y = data
                        x = x.cuda()
                        y = y.cuda()
                        y = y.to(torch.float32)
                        output = module(x)
                        loss = loss_fn(output, y)
                        # loss = loss_fn(output, y)
                        epoch_test_loss += loss.item() * batch_size
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
                SEN = -1 if TP + FN == 0 else TP / (TP + FN)  # 灵敏度
                SPE = -1 if TN + FP == 0 else TN / (TN + FP)  # 特异性
                PPV = -1 if TP + FP == 0 else TP / (TP + FP)  # 正预测率
                NPV = -1 if TN + FN == 0 else TN / (TN + FN)  # 负预测率
                F1 = -1 if PPV + SEN == 0 else 2 * (PPV * SEN) / (PPV + SEN)  # F1

                SEN_list.append(SEN)
                SPE_list.append(SPE)
                PPV_list.append(PPV)
                NPV_list.append(NPV)
                F1_list.append(F1)

                test_acc_list.append(ACC)
                test_loss_list.append(float(epoch_test_loss))
                if ACC > best_acc:
                    best_acc = ACC

                AUC,FPR,TPR = roc.GetROC(torch.as_tensor(Y_train), torch.as_tensor(Y_pred), epoch_i + 1, k + 1)
                AUC_list.append(AUC)
                FPR_list.append(FPR)
                TPR_list.append(TPR)


                # if epoch_i + 1 == 50:
                #     torch.save(module.state_dict(), f"../pretrain_module/pretrain_{k + 1}.pt")

                p_table.add_row(
                    [institution,
                     epoch_i + 1,
                     format(float(train_loss_list[-1]), '.3f'),
                     format(float(train_acc_list[-1]), '.4f'),
                     format(float(test_loss_list[-1]), '.3f'),
                     format(float(test_acc_list[-1]), '.4f'),
                     format(float(best_acc), '.4f'),
                     format(float(lr * 1e4), '.4f'),
                     format(float(AUC), '.3f'),
                     format(float(time.time() - last_time), '.2f')])

                if epoch_i > 3 and train_loss_list[-2] - train_loss_list[-1] > 3:
                    lr = lr * decay
                    print("lr has changed to ", lr)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                last_time = time.time()
                print(p_table)
                if EarlyStop:
                    if epoch_i >= EarlyStop_epoch:
                        lr = early_stop(epoch_test_loss, module, lr)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

            best_acc_list.append(best_acc)
            print("最优准确率：", best_acc_list[-1])

            draw_result_pic([train_acc_list, test_acc_list], 0, f'{k + 1}Fold_acc')
            draw_result_pic([train_loss_list, test_loss_list], 0, f'{k + 1}Fold_loss')
            roc.DrawROC(f'{institution}')

            # 记录每一折的数据，是一个二维的列表
            train_acc_list_kf.append(train_acc_list)
            train_loss_list_kf.append(train_loss_list)
            test_acc_list_kf.append(test_acc_list)
            test_loss_list_kf.append(test_loss_list)

            SEN_list_kf.append(SEN_list)
            SPE_list_kf.append(SPE_list)
            PPV_list_kf.append(PPV_list)
            NPV_list_kf.append(NPV_list)
            F1_list_kf.append(F1_list)
            AUC_list_kf.append(AUC_list)
            FPR_list_kf.append(FPR_list)
            TPR_list_kf.append(TPR_list)

            k += 1
            SaveArgsInfo()

        avg_train_acc = train_acc_list_kf
        avg_train_loss = train_loss_list_kf
        avg_test_acc = test_acc_list_kf
        avg_test_loss = test_loss_list_kf

        avg_sen = SEN_list_kf
        avg_spe = SPE_list_kf
        avg_ppv = PPV_list_kf
        avg_npv = NPV_list_kf
        avg_f1 = F1_list_kf
        avg_auc = AUC_list_kf

        res = pd.DataFrame()
        res['训练集准确率'] = avg_train_acc
        res['测试集准确率'] = avg_test_acc
        res['训练集损失值'] = avg_train_loss
        res['测试集损失值'] = avg_test_loss
        res['灵敏度'] = avg_sen
        res['特异性'] = avg_spe
        res['正预测率'] = avg_ppv
        res['负预测率'] = avg_npv
        res['F1分数'] = avg_f1
        res['AUC'] = avg_auc
        res['FPR'] = FPR_list_kf
        res['TPR'] = TPR_list_kf
        res.to_csv(f"../result/result_{institution}.csv", encoding='utf_8_sig')

        # 传结果list格式： [train(list), test(list)]
        # draw_result_pic(res=[avg_train_acc, avg_test_acc],
        #                 start_epoch=0,
        #                 pic_title='acc')
        # draw_result_pic(res=[avg_train_loss, avg_test_loss],
        #                 start_epoch=0,
        #                 pic_title='loss')


if __name__ == '__main__':
    if not os.path.exists("../result"):
        os.makedirs("../result")
    if not os.path.exists("../pretrain_module"):
        os.makedirs("../pretrain_module")
    if not os.path.exists("../result/pic"):
        os.makedirs("../result/pic")

    start_time = time.time()
    Train()
    end_time = time.time()
    spend_time = end_time - start_time
    min = int(spend_time // 60)
    sec = int(spend_time - 60 * min)
    print(f"模型训练总用时:{min}分{sec}秒")
