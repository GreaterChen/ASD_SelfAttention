
from requirements import *
from args import *


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

        data = pd.read_csv("../description/kendall_sort.csv")
        index = np.array(data.iloc[:kendall_nums, 0])
        for file in tqdm(self.files, desc='Datasets', file=sys.stdout):
            file_path = root_path + "/" + file
            if 'csv' in root_path:
                if fisher_r2z:
                    self.data.append(torch.as_tensor(np.arctanh(pd.read_csv(file_path).iloc[:, index].values)))
                else:
                    self.data.append(torch.as_tensor(pd.read_csv(file_path).values))  # 转化为tensor类型
            elif 'pkl' in root_path:
                if fisher_r2z:
                    self.data.append(torch.as_tensor(np.arctanh(pd.read_pickle(file_path).iloc[:, index].values)))
                else:
                    self.data.append(torch.as_tensor(pd.read_pickle(file_path).iloc[:, index].values))

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

            if self.counter % 5 == 0:
                lr_c = lr * 0.2
                print("lr is changed from", lr, "to", lr_c)

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


def Draw_ROC(Y_train, Y_pred, desc, fold):
    fpr, tpr, thresholds_keras = roc_curve(Y_train, Y_pred)
    AUC = auc(fpr, tpr)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC={:.3f}".format(AUC))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{fold}-{desc} ROC curve')
    plt.legend()
    plt.savefig(f"../result/pic/{fold}_roc.png", dpi=500)
    plt.show()

    return AUC


def GetAvg(a):
    max_len = 0
    for item in a:
        max_len = len(item) if len(item) > max_len else max_len

    avg = []
    for j in range(max_len):
        sum = 0
        cnt = 0
        for i in range(5):
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