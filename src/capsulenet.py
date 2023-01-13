from requirements import *
from capsulelayers import DenseCapsule


class CapsuleNet(nn.Module):
    """
    A Capsule Network
    :param input_size: data size = [channels, width, height]
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes)
    """

    def __init__(self, input_size, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.routings = routings
        self.digitcaps = DenseCapsule(in_num_caps=input_size[1], in_dim_caps=input_size[2],
                                      out_num_caps=2, out_dim_caps=16, routings=routings)

        self.l1 = nn.Linear(32, 10)
        self.r1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(10, 2)
        self.s = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.digitcaps(x)  # (2,2,16)
        length = x.norm(dim=-1)  # (2,2)
        y = x.reshape(x.shape[0], -1)  # (2,32)
        y = self.l1(y)
        y = self.r1(y)
        y = self.l2(y)
        y = self.s(y)

        return length, y


def caps_loss(y_true, y_pred):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    return L_margin
