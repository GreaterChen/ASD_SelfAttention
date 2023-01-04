from Structure import *


class Module(Structure):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, x):
        return self.FC(x)
