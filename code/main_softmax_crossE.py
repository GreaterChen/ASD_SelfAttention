from requirements import *
from args import *
from utils import draw_result_pic, EarlyStopping, GetAvg, GetData
from Module import Module


def Train():
    all_data = GetData(root_path, label_path, dataset_size)  # 一次性读取所有数据

    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 初始化5折交叉验证的工具

    # 对每一折进行记录
    train_acc_list_kf = []
    train_loss_list_kf = []
    test_acc_list_kf = []
    test_loss_list_kf = []

    split_range = 0
    last_time = time.time()
    k = 0  # 表征第几折
    for train_index, test_index in kf.split(all_data):  # 此处获取每一折的索引
        # 对于每一折来说，都要从0开始训练模型
        # 因为如果不同折训练同一个模型，会出现当前折的测试集曾被另一折当作训练集训练，导致准确率异常
        module = Module()
        module = module.cuda()

        # 损失函数：交叉熵
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.cuda()
        # 优化器：SGD
        optimizer = torch.optim.SGD(module.parameters(), lr=learn_rate)

        early_stop = EarlyStopping()

        p_table = PrettyTable(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "time(s)"])

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

        # 下面开始当前折的训练
        for epoch_i in range(epoch):
            if early_stop.early_stop:
                break
            # 对该折该轮的所有dataloader进行记录
            epoch_train_loss = 0
            epoch_train_acc = 0
            epoch_test_loss = 0
            epoch_test_acc = 0

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

                output = module(x)

                loss = loss_fn(output, y)
                epoch_train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.step()
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
                        if res[0] > res[1]:
                            test_acc = test_acc + 1 if y[i][0] > y[i][1] else test_acc
                        if res[0] < res[1]:
                            test_acc = test_acc + 1 if y[i][0] < y[i][1] else test_acc
                    epoch_test_acc += test_acc

            test_acc_list.append(float(epoch_test_acc / test_size))
            test_loss_list.append(float(epoch_test_loss))

            p_table.add_row(
                [epoch_i + 1, float(train_loss_list[-1]), float(train_acc_list[-1]), float(test_loss_list[-1]),
                 float(test_acc_list[-1]), time.time() - last_time])
            last_time = time.time()
            print(p_table)
            if epoch_i >= 30:
                early_stop(epoch_test_loss, module)

        draw_result_pic([train_acc_list, test_acc_list], 0, f'{k + 1}Fold_acc')
        draw_result_pic([train_loss_list, test_loss_list], 0, f'{k + 1}Fold_loss')

        # 记录每一折的数据，是一个二维的列表
        train_acc_list_kf.append(train_acc_list)
        train_loss_list_kf.append(train_loss_list)
        test_acc_list_kf.append(test_acc_list)
        test_loss_list_kf.append(test_loss_list)

        K_Fold_res = pd.DataFrame()
        K_Fold_res['训练集损失值'] = train_loss_list_kf[k]
        K_Fold_res['训练集准确率'] = train_acc_list_kf[k]
        K_Fold_res['测试集损失值'] = test_loss_list_kf[k]
        K_Fold_res['测试集准确率'] = test_acc_list_kf[k]
        K_Fold_res.to_csv(f"../result/{k + 1}_Fold.csv")
        k += 1

    avg_train_acc = GetAvg(train_acc_list_kf)
    avg_train_loss = GetAvg(train_loss_list_kf)
    avg_test_acc = GetAvg(test_acc_list_kf)
    avg_test_loss = GetAvg(test_loss_list_kf)

    res = pd.DataFrame()
    res['训练集准确率'] = avg_train_acc
    res['测试集准确率'] = avg_test_acc
    res['训练集损失值'] = avg_train_loss
    res['测试集损失值'] = avg_test_loss
    res.to_csv("../result/res.csv")
    print(res)

    # 传结果list格式： [train(list), test(list)]
    draw_result_pic(res=[avg_train_acc, avg_test_acc],
                    start_epoch=0,
                    pic_title='acc')
    draw_result_pic(res=[avg_train_loss, avg_test_loss],
                    start_epoch=0,
                    pic_title='loss')


if __name__ == '__main__':
    start_time = time.time()
    Train()
    end_time = time.time()
    spend_time = end_time - start_time
    min = spend_time // 60
    sec = spend_time - 60 * min
    print(f"模型训练总用时:{min}分{sec}秒")