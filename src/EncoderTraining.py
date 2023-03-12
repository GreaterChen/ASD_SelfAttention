import sys

import torch.utils.data

from requirements import *
from args import *
from utils import *
from SparseAutoEncoder import *


# 计算平均激活值rho_bar
def cal_rho_bar(rho):
    rho_i = F.softmax(rho, dim=0)
    return torch.sum(rho_i, dim=0) / batch_size


# 计算rho_bar和rho的KL散度
def KL(rho, rho_bar):
    k1 = torch.sum(rho * torch.log(rho / rho_bar))
    k2 = torch.sum((1 - rho) * torch.log((1 - rho) / (1 - rho_bar)))
    return k1 + k2


rho = 0.05  # 稀疏系数
beta = 1  # KL散度权重
if __name__ == '__main__':
    all_data = GetData(root_path, label_path, dataset_size)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    tb = PrettyTable()
    tb.field_names = ["Fold", "Epoch", "train_loss", "test_loss"]

    k = 0
    for train_index, test_index in kf.split(all_data):

        # model
        encoder1 = AutoEncoder(input_dim=Vector_len, output_dim=1000)
        encoder2 = AutoEncoder(input_dim=1000, output_dim=600)
        decoder1 = AutoEncoder(input_dim=600, output_dim=1000)
        decoder2 = AutoEncoder(input_dim=1000, output_dim=Vector_len)
        encoders = [encoder1, encoder2, decoder1, decoder2]

        model = StackAutoEncoder(encoders)
        model.cuda()

        # fold_data
        train_fold_data = Subset(all_data, train_index)
        test_fold_data = Subset(all_data, test_index)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

        rho_tensor = torch.FloatTensor([rho for _ in range(600)])
        rho_tensor = rho_tensor.cuda()

        for e in range(epoch):
            train_total_loss = 0
            test_total_loss = 0

            # train
            model.train()
            for data in tqdm(train_fold_data, desc=f"Fold:{k} Epoch:{e} Training", file=sys.stdout):
                x, _ = data
                x = x.cuda()
                x = x.to(torch.float32)

                encoder_out, decoder_out = model(x)

                # loss + KL
                loss = loss_fn(decoder_out, x)
                rho_bar = cal_rho_bar(encoder_out)
                loss += beta * KL(rho_tensor, rho_bar)
                train_total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test
            model.eval()
            for data in tqdm(test_fold_data, desc=f"Fold:{k} Epoch:{e} Testing", file=sys.stdout):
                x, _ = data
                x = x.cuda()
                x = x.to(torch.float32)

                encoder_out, decoder_out = model(x)

                loss = loss_fn(decoder_out, x)
                rho_bar = cal_rho_bar(encoder_out)
                loss += beta * KL(rho_tensor, rho_bar)
                test_total_loss += loss

            tb.add_row([k + 1, e + 1, format(float(train_total_loss), ".3f"), format(float(test_total_loss), ".3f")])
            print(tb)
        k += 1
        torch.save(model, "SAE.pth")
        break
        print(tb)
    pass
