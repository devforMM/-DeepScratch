import torch

class Relu:
    def __init__(self):
        ...
    def forward(self,x):
        return torch.maximum(torch.tensor(0.0), x)
class LeakyRelu:
    def __init__(self,beta):
        self.beta=beta
    def forward(self,x):
        return torch.maximum(torch.tensor(self.beta),x)
    
class Sigmoid:
    def __init__(self):
        ...
    def forward(self,x):
            return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
class Tanh:
    def __init__(self):
        ...
    def forward(self,x):
        return torch.tanh(x)

