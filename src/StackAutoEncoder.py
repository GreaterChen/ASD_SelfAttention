from requirements import *


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, train_layer=False):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_layer = train_layer  # 是否逐层训练
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.Sigmoid()
        )
        # 译码器
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        # 若为逐层训练，则译码输出
        if self.train_layer:
            return self.decoder(out)
        else:
            return out

    def lock_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def require_grad(self):
        for p in self.parameters():
            p.requires_grad = True


class StackAutoEncoder(nn.Module):

    def __init__(self, encoders):
        super(StackAutoEncoder, self).__init__()
        self.encoders = encoders

        for encoder in encoders:
            encoder.train_layer = False

        self.encoder1 = encoders[0]
        self.encoder2 = encoders[1]
        self.encoder3 = encoders[2]
        self.encoder4 = encoders[3]

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        return out
