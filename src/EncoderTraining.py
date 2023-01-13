import torch.utils.data

from utils import *
from StackAutoEncoder import *
from args import *


def train_layer(train_data, encoders, layer_num):
    loss_list = []
    print("=" * 50 + "Training Layer{}".format(layer_num) + "=" * 50)
    for encoder in encoders:
        encoder.cuda()

    # MSE
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()

    # SGD
    optimizer = torch.optim.SGD(encoders[layer_num].parameters(), lr=learn_rate)

    for e in range(epoch):
        total_loss = 0

        # 固定前layer_num层参数
        if layer_num != 0:
            for i in range(layer_num):
                encoders[i].lock_grad()
                encoders[i].train_layer = False

        for data in tqdm(train_data, desc=f"Epoch:{e} Training:", file=sys.stdout):
            x, _ = data
            x = x.cuda()
            x = x.to(torch.float32)
            out = x

            # 前layer_num层进行前向计算
            if layer_num != 0:
                for i in range(layer_num):
                    out = encoders[i](out)

            # train
            pred = encoders[layer_num](out)

            optimizer.zero_grad()
            loss = loss_fn(pred, out)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Epoch:{} Loss:{}".format(e, total_loss))
        loss_list.append(total_loss)
    return loss_list


def train_model(train_data, test_data, model):
    train_loss_list = []
    test_loss_list = []
    print("=" * 50 + "Training Model" + "=" * 50)
    model.cuda()

    # 解锁参数
    for param in model.parameters():
        param.require_grad = True

    # SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    # MSE
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()

    for e in range(epoch):
        train_total_loss = 0
        test_total_loss = 0

        # train
        model.train()
        for data in tqdm(train_data, desc=f"Epoch:{e} Training:", file=sys.stdout):
            x, _ = data
            x = x.cuda()
            x = x.to(torch.float32)

            out = model(x)

            optimizer.zero_grad()
            loss = loss_fn(out, x)
            train_total_loss += loss
            loss.backward()
            optimizer.step()
        print("Epoch:{} Loss:{}".format(e, train_total_loss))
        train_loss_list.append(train_total_loss)

        # test
        model.eval()
        for data in tqdm(test_data, desc=f"Epoch:{e} Testing:", file=sys.stdout):
            x, _ = data
            x = x.cuda()
            x = x.to(torch.float32)

            out = model(x)
            loss = loss_fn(out, x)
            test_total_loss += loss
        print("Epoch:{} Loss:{}".format(e, test_total_loss))
        test_loss_list.append(test_total_loss)
    return train_loss_list, test_loss_list


if __name__ == '__main__':
    all_data = GetData(root_path, label_path, dataset_size)

    # test and train split
    train_prop = 0.8
    train_size = int(train_prop * len(all_data))
    test_size = len(all_data) - train_size

    # data
    train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])

    # encoders
    encoder1 = AutoEncoder(Vector_len, 4000, train_layer=True)
    encoder2 = AutoEncoder(4000, 2000, train_layer=True)
    decoder1 = AutoEncoder(2000, 4000, train_layer=True)
    decoder2 = AutoEncoder(4000, Vector_len, train_layer=True)
    encoders = [encoder1, encoder2, decoder1, decoder2]

    # train_layer
    for layer_num in range(len(encoders)):
        res = train_layer(train_data, encoders, layer_num)
        plt.figure()
        plt.plot(res)
        plt.show()

    # train_model
    model = StackAutoEncoder(encoders=encoders)
    train_res, test_res = train_model(train_data, test_data, model)
    plt.figure()
    plt.plot(train_res)
    plt.show()
    plt.figure()
    plt.plot(test_res)
    plt.show()

    pass
