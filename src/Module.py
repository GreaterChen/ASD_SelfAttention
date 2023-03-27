from Structure import *


class Module(Structure):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, x):
        log.info("Module模块输入 x", x)
        return self.attention_with_ffn_and_ln(x)
        # return self.CNN(x)
        # return self.LSTM(x)
        # return self.CLAttention(x)
        # return self.FCN(x)
