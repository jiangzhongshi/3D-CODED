from torch.nn.modules.module import Module
from functions.nnd import NNDFunction

class NNDModule(Module):
    def forward(self, input1, input2):
        mydist = NNDFunction()
        dist1, dist2 = mydist(input1, input2)
        self.idx1 = mydist.idx1
        self.idx2 = mydist.idx2
        return dist1, dist2
