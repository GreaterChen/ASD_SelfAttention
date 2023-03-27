from requirements import *


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
