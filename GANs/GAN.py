import sys
sys.path.append("../")
from core.MLp_layer import Layer
import torch
class Generator:
    def __init__(self):
        self.layers=[]
        self.weights=[]
    def  add_layer(self,input_shape,n_neurones,initializer):
        l=Layer(input_shape,n_neurones,initializer)
        self.layers.append(l)
        self.weights.append(l.w)
    def forward(self,random_noise):
        for l in self.layers:
            random_noise=l.forward(random_noise)
        return random_noise


class Discriminator:
    def __init__(self):
        self.layers=[]
        self.weights=[]
    def add_layer(self,input_shape,n_neurones,initializer):
        l=Layer(input_shape,n_neurones,initializer)
        self.layers.append(l)
        self.weights.append(l.w)
    def forward(self,real_x):
        for l in self.layers:
            real_x=l.forward(real_x)
        return real_x



class Generator_loss:
    @staticmethod
    def compute(Y_fake):
        fake_prob=torch.sigmoid(Y_fake)
        return - torch.mean(torch.log(fake_prob + 1e-8))
    
class Discriminator_loss:
     @staticmethod
     def compute(Y_true,Y_fake):
        fake_prob=torch.sigmoid(Y_fake)
        true_prob=torch.sigmoid(Y_true)
        return -torch.mean(torch.log(true_prob + 1e-8)+torch.log(1-fake_prob + 1e-8))