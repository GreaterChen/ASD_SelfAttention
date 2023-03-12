import pandas as pd

from requirements import *
from args import *

eps = 1e-5


class GetData(Dataset):
    def __init__(self, root_path, label_path, dataset_size):
        """
        继承于Dataset,与主函数中DataLoader一起取数据
        :param root_path: 数据集目录地址
        :param label_path: 标签文件地址
        :param dataset_size: 训练的样本总数,-1代表全部训练
        """
        self.data = []
        self.label = []
        self.files = os.listdir(root_path)
        self.files.sort()  # 排一下序，确保和标签是对准的
        self.label_info = pd.read_csv(label_path)

        self.files = self.files[:dataset_size] if dataset_size != -1 else self.files
        self.label_info = self.label_info[:dataset_size] if dataset_size != -1 else self.label_info

        if not CheckOrder(self.files, self.label_info):
            exit()

        data = pd.read_csv("../description/Kendall_sort_all.csv")
        index = np.array(data.iloc[:kendall_nums, 1])

        for file in tqdm(self.files, desc='Datasets', file=sys.stdout):
            file_path = root_path + "/" + file
            if static:
                data = torch.as_tensor(pd.read_pickle(file_path).values)
                data = data.reshape(1, data.shape[0], data.shape[1])
                self.data.append(data)
            elif fisher_r2z:
                self.data.append(torch.as_tensor(np.arctanh(pd.read_pickle(file_path).iloc[:, index].values)))
            else:
                temp = pd.read_pickle(file_path).iloc[:, index].values
                self.data.append(torch.as_tensor(temp))

        label = list(zip(self.label_info.group_1.values, self.label_info.group_2.values))

        self.label = torch.tensor(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def draw_result_pic(res: list, start_epoch: int, pic_title: str):
    x = [idx for idx in range(len(res[0]))]
    y0 = res[0]  # train
    y1 = res[1]  # test

    if 'acc' in pic_title:
        plt.figure()
        plt.plot(x[start_epoch:], y0[start_epoch:], label='train', c='blue')
        plt.plot(x[start_epoch:], y1[start_epoch:], label='test', c='red')
        plt.xlabel('epoch')
        plt.xticks([i * 10 for i in range(len(y0) // 10)])
        plt.title(pic_title)
        plt.legend()
        if 'Fold' not in pic_title:
            plt.savefig(f'../result/pic/{pic_title}.png', dpi=500, bbox_inches='tight')
        else:
            plt.savefig(f'../result/pic/{pic_title}.png', dpi=500, bbox_inches='tight')
        plt.show()
    elif 'loss' in pic_title:
        plt.figure()
        plt.plot(x[start_epoch:], y0[start_epoch:], label='train', c='blue')
        plt.xlabel('epoch')
        plt.xticks([i * 10 for i in range(len(y0) // 10)])
        plt.title('train' + pic_title)
        plt.legend()
        if 'Fold' not in pic_title:
            plt.savefig(f'../result/pic/train_{pic_title}.png', dpi=500, bbox_inches='tight')
        else:
            plt.savefig(f'../result/pic/train_{pic_title}.png', dpi=500, bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(x[start_epoch:], y1[start_epoch:], label='test', c='blue')
        plt.xlabel('epoch')
        plt.xticks([i * 10 for i in range(len(y0) // 10)])
        plt.title('test' + pic_title)
        plt.legend()
        if 'Fold' not in pic_title:
            plt.savefig(f'../result/pic/test_{pic_title}.png', dpi=500, bbox_inches='tight')
        else:
            plt.savefig(f'../result/pic/test_{pic_title}.png', dpi=500, bbox_inches='tight')
        plt.show()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_loss, model, lr):

        score = -val_loss

        lr_c = lr

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            # if self.counter % 10 == 0:
            #     lr_c = lr * decay
            #     print("lr is changed from", lr, "to", lr_c)

            if self.counter >= self.patience:
                self.early_stop = True
                torch.save(self.best_model, self.path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return lr_c

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model = model.state_dict()
        self.val_loss_min = val_loss


class ROC:
    def __init__(self):
        self.best_fpr = None
        self.best_tpr = None
        self.best_auc = 0
        self.epoch = None
        self.fold = None

    def GetROC(self, Y_train, Y_pred, epoch, fold):
        fpr, tpr, thresholds_keras = roc_curve(Y_train, Y_pred)
        AUC = auc(fpr, tpr)
        if AUC > self.best_auc:
            self.epoch = epoch
            self.fold = fold
            self.best_fpr = fpr
            self.best_tpr = tpr
            self.best_auc = AUC
        return AUC

    def DrawROC(self, info=''):
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self.best_fpr, self.best_tpr, label="AUC={:.3f}".format(self.best_auc))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'{self.fold}-{self.epoch} ROC curve')
        plt.legend()
        plt.savefig(f"../result/pic/{self.fold}_roc{info}.png", dpi=500)
        plt.show()


def GetAvg(a):
    max_len = 0
    for item in a:
        max_len = len(item) if len(item) > max_len else max_len

    avg = []
    for j in range(max_len):
        sum = 0
        cnt = 0
        for i in range(k_fold):
            if j < len(a[i]):
                sum += a[i][j]
                cnt += 1
        if cnt != 0:
            avg.append(sum / cnt)
    return avg


def CheckOrder(files, label):
    label_name = label['SUB_ID'].to_list()
    error_sign = 1
    for i in range(len(label_name)):
        if str(label_name[i]) not in str(files[i]):
            error_sign = 0
            break

    if error_sign:
        print("数据、标签已对准")
        return True
    else:
        print("数据、标签未对准！请暂停检查！")
        return False


def SaveArgsInfo():
    args = []
    value = [dataset_size, batch_size, Head_num, epoch, learn_rate, dropout, ffn_hidden_mult, begin_fold, end_fold,
             L1_en,
             L1_weight_decay, L2_en, L2_weight_decay, fisher_r2z, kendall, kendall_nums, pin_memory, num_workers,
             pre_train, EarlyStop, EarlyStop_patience, EarlyStop_epoch, Windows_num, Vector_len]

    args.append("dataset_size")
    args.append("batch_size")
    args.append("Head_num")
    args.append("epoch")
    args.append("learn_rate")
    args.append("dropout")
    args.append("ffn_hidden_mult")
    args.append("begin_fold")
    args.append("end_fold")
    args.append("L1_en")
    args.append("L1_weight_decay")
    args.append("L2_en")
    args.append("L2_weight_decay")
    args.append("fisher_r2z")
    args.append("kendall")
    args.append("kendall_nums")
    args.append("pin_memory")
    args.append("num_workers")
    args.append("pre_train")
    args.append("EarlyStop")
    args.append("EarlyStop_patience")
    args.append("EarlyStop_epoch")
    args.append("Windows_num")
    args.append("Vector_len")

    desc = pd.DataFrame()
    desc["args"] = args
    desc["value"] = value
    desc.to_csv("../result/Args.csv")


def L1_decay(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    return regularization_loss


class dot_loss(nn.Module):
    def __init__(self):
        super(dot_loss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = - torch.sum(y_pred * y_true, dim=1)  # batch_size X 1
        return loss.mean()


class WeightedFocalLoss(nn.Module):
    """Non weighted version of Focal Loss"""

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        at = at.view(batch_size, 2)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def ZCAWhite(temp):
    sigma = temp.dot(temp.T) / temp.shape[1]
    [u, s, v] = np.linalg.svd(sigma)
    temp = u.dot(np.diag(1. / np.sqrt(s + eps))).dot(u.T.dot(temp))
    return temp


def PCA_test(data):
    print(data.shape)
    sigma = data.dot(data.T) / data.shape[1]
    [u, s, v] = np.linalg.svd(sigma)
    xRot = u.T.dot(data)
    print(u.shape)
    xRot[0:1024, :] = u[:, 0:1024].T.dot(data)
    print(xRot.shape)
    return xRot[0:1024, :]


def PCAWhite(data):
    sigma = data.dot(data.T) / data.shape[1]
    [u, s, v] = np.linalg.svd(sigma)
    xPCAWhite = np.diag(1. / np.sqrt(s + eps)).dot(u.T.dot(data))
    print(xPCAWhite.shape)
    return xPCAWhite


class PrototypeLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.asd = []  # [label,[xi,xj]]
        self.hc = []
        self.pro_asd = []
        self.pro_hc = []
        self.lamda = 1
        self.eps = 1e-1
        self.alpha = 1e-3
        self.init_prototypes()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def init_prototypes(self):
        self.asd.append([1, np.array([1, 0])])
        self.hc.append([0, np.array([0, 1])])

    def forward(self, output, y):
        output = self.softmax(output)
        print(len(self.asd) + len(self.hc))
        self.pro_hc.clear()
        self.pro_asd.clear()
        result = []
        min_distance = torch.tensor(1e9, dtype=torch.float32).cuda()
        min_item = None
        min_index = -1
        ploss = torch.tensor(1e9, dtype=torch.float32, requires_grad=True).cuda()  # 原型损失
        total_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        for i in range(batch_size):
            pos = output[i]  # [xi,xj]
            for j, item in enumerate(self.asd):  # 对于所有asd类别的原型
                distance = ((pos[0] - item[1][0]) ** 2 + (pos[1] - item[1][1]) ** 2).sqrt()  # 欧氏距离
                self.pro_asd.append(torch.exp(-1 * self.lamda * distance))  # 准备做softmax
                if distance < min_distance:  # 记录距离最近的原型
                    min_distance = distance
                    min_item = item
                    min_index = j

                if torch.abs(y[i][0] - 1) < self.eps:  # 其实都是整数，但是因为使用浮点数存储，所以用此办法判断一致
                    ploss = torch.min(ploss, distance)

            for j, item in enumerate(self.hc):
                distance = ((pos[0] - item[1][0]) ** 2 + (pos[1] - item[1][1]) ** 2).sqrt()
                self.pro_hc.append(torch.exp(-1 * self.lamda * distance))
                if distance < min_distance:
                    min_distance = distance
                    min_item = item
                    min_index = j

                if torch.abs(y[i][0] - 0) < self.eps:
                    ploss = min(ploss, distance)
            if self.model.training:
                if abs(min_item[0] - y[i][0]) < self.eps:  # 如果距离最近的原型和当前样本类别相同
                    new_pos = np.array((min_item[1] + pos.tolist())) / 2  # 取均值
                    if abs(y[i][0] - 1) < self.eps:  # 如果原型来自asd
                        self.asd.pop(min_index)
                        self.asd.append([y[i][0], new_pos])
                    else:
                        self.hc.pop(min_index)
                        self.hc.append([y[i][0], new_pos])
                else:  # 如果距离最近的原型和当前样本类别不同
                    if abs(y[i][0] - 1) < self.eps:  # 如果属于asd
                        self.asd.append([int(y[i][0]), np.array(pos.tolist())])
                    else:
                        self.hc.append([int(y[i][0]), np.array(pos.tolist())])

            pro_asd_sum = torch.sum(torch.as_tensor(self.pro_asd))
            pro_hc_sum = torch.sum(torch.as_tensor(self.pro_hc))

            belongs_asd = pro_asd_sum / (pro_asd_sum + pro_hc_sum)
            belongs_hc = pro_hc_sum / (pro_asd_sum + pro_hc_sum)
            result.append([belongs_asd, belongs_hc])
            total_loss = total_loss + -1 * (y[i][0] * torch.log(belongs_asd) + (1 - y[i][0]) * torch.log(belongs_hc))
            # total_loss = total_loss + -1 * torch.log(belongs_asd)
        return total_loss + ploss, result


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=116):
        # d_model=1024,dropout=0.1,
        # max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
        # 一般100或者200足够了。
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # (116,1024)矩阵，保持每个位置的位置编码，一共116个位置，
        # 每个位置用一个512维度向量来表示其位置编码
        position = torch.arange(0, max_len).unsqueeze(1)
        # (116) -> (116,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(torch.tensor(10000.0)) / d_model))
        # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        pe = pe.unsqueeze(0)
        # (116, 1024) -> (1, 116, 1024) 为batch.size留出位置
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 接受1.Embeddings的词嵌入结果x，
        # 然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        # 例如，假设x是(30,10,1024)的一个tensor，
        # 30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        # 则该行代码的第二项是(1, min(10, 116), 1024)=(1,10,1024)，
        # 在具体相加的时候，会扩展(1,10,1024)为(30,10,1024)，
        # 保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
        return self.dropout(x)  # 增加一次dropout操作


class Logger:
    """
    打印变量信息：原始变量 || 变量形状
    display 是否打印
    ori_var 是否打印原始变量信息
    """

    def __init__(self, display=True, ori_var=False):
        self.display = display
        self.ori = ori_var
        self.prefix = "-" * 30
        self.colorpad = {
            "INFO": ['\033[1;36m', '\033[0m'],
            "WARNING": ['\033[1;31m', '\033[0m']
        }

    def separator(self):
        if self.display:
            print('\033[5;33m' + self.prefix * 4 + '\033[0m')

    def info(self, msg, var=None):
        if self.display:
            print("")
            print(self.colorpad["WARNING"][0] + self.prefix + msg + self.prefix + self.colorpad["WARNING"][1])
            try:
                print(self.colorpad["INFO"][0] + "----- shape -----" + self.colorpad["INFO"][1])
                print(var.shape)
            except Exception as e:
                pass
            if self.ori:
                print(self.colorpad["INFO"][0] + "----- variable -----" + self.colorpad["INFO"][1])
                print(var)
