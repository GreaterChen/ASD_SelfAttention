from requirements import *
from args import *
from utils import *
import torch.utils.data
from Regularization import *


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out

def KL_devergence(p, q):
    q = F.softmax(q, dim=0)
    q = torch.sum(q, dim=0) / batch_size  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


rho = 0.1  # 稀疏系数
beta = 1  # KL散度权重
if __name__ == '__main__':

    p_table = PrettyTable(
        ["Epoch", "train_loss", "test_loss"]
    )
    all_data = GetData(root_path, label_path, dataset_size)

    # test and train split
    train_prop = 0.8
    train_size = int(train_prop * len(all_data))
    test_size = len(all_data) - train_size

    # data
    train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)

    model = AutoEncoder(input_dim=kendall_nums, output_dim=sae_hidden_nums)
    model = model.cuda()

    loss_fn = nn.MSELoss()
    L1_reg_loss = Regularization(model, L1_weight_decay, p=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    rho_tensor = torch.FloatTensor([rho for _ in range(sae_hidden_nums)])
    rho_tensor = rho_tensor.cuda()

    for e in tqdm(range(epoch), desc="running", file=sys.stdout):
        train_total_loss = 0
        test_total_loss = 0

        # train
        model.train()
        for data in train_loader:
            x, _ = data
            x = x.cuda()
            x = x.to(torch.float32)

            encoder_out, decoder_out = model(x)

            loss = loss_fn(decoder_out, x)
            # loss += L1_reg_loss(model)
            # loss += beta * KL_devergence(rho_tensor, encoder_out)
            train_total_loss += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test
        model.eval()
        for data in test_loader:
            x, _ = data
            x = x.cuda()
            x = x.to(torch.float32)

            encoder_out, decoder_out = model(x)

            loss = loss_fn(decoder_out, x)
            # loss += beta * KL_devergence(rho_tensor, encoder_out)
            test_total_loss += loss.item()
        p_table.add_row([e + 1, format(float(train_total_loss), ".3f"), format(float(test_total_loss), ".3f")])
        print("\n", p_table)

    torch.save(model, "SAE.pth")
