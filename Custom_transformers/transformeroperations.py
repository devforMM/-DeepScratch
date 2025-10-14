import torch
import sys
sys.path.append("../")
from core.MLp_initializers import *
xavier=XavierNormal()

class Positional_EmbeddingLayer:
    def __init__(self,vocab_size,dmodel):
        self.dmodel=dmodel
        self.vocab_size=vocab_size
        self.weights=torch.randn(1,1,vocab_size+1,dmodel,requires_grad=True)
    def get_postinal_embedings(self,x):
        embedings= x.unsqueeze(-1)* self.weights
        return pos_encoding(embedings,self.dmodel)

class VitELayer:
    def __init__(self,vocab_size,dmodel):
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.embedigns=xavier.initialize(vocab_size,dmodel)
    def get_postional_embedings(self,x):
        embedings=x.float()@self.embedigns
        return pos_encoding(embedings,self.dmodel)
    
class ELayer:
    def __init__(self,vocab_size,dmodel):
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.embedigns=torch.randn(vocab_size,dmodel,requires_grad=True)
    def get_postional_embedings(self,x):
        embedings=self.embedigns[x]
        return pos_encoding(embedings,self.dmodel)


class AddNormLayer:
    def __init__(self, d_model, eps=1e-6):
        self.weights=[]
        self.gamma = torch.ones(d_model, requires_grad=True)
        self.beta = torch.zeros(d_model, requires_grad=True)
        self.eps = eps
        self.weights.extend([self.gamma,self.beta])
    def forward(self,x,y):
        y =x+y
        mean = y.mean(dim=-1, keepdim=True)
        var = y.var(dim=-1, keepdim=True)

        norm_y = (y - mean) / torch.sqrt(var + self.eps)
        norm_y = self.gamma * norm_y + self.beta
        return norm_y

class Feed_Forward:
    def __init__(self,dmodel):
        self.d_model=dmodel
        dff=512*self.d_model
        self.w1=torch.randn(self.d_model,dff,requires_grad=True)
        self.w2=torch.randn(dff,self.d_model,requires_grad=True)
        self.b1=torch.randn(dff,requires_grad=True)
        self.b2=torch.randn(self.d_model,requires_grad=True)
        self.weights=[]
        self.weights.extend([self.w1,self.w2,self.b1,self.b2])
    def forward(self,x):
        y=x@self.w1+self.b1    
        return torch.relu(y)@self.w2+self.b2

class Linear:
    def __init__(self,dmodel,vocab_size):
        self.weights=[]
        self.w=torch.randn(dmodel,vocab_size,requires_grad=True)
        self.b=torch.randn(vocab_size,requires_grad=True)
        self.weights.extend([self.w,self.b])
    def forward(self,x):
        return x@self.w+self.b

class Positional_EmbeddingLayer:
    def __init__(self,vocab_size,dmodel):
        self.dmodel=dmodel
        self.vocab_size=vocab_size
        self.weights=torch.randn(1,1,vocab_size+1,dmodel,requires_grad=True)
    def get_postinal_embedings(self,x):
        embedings= x.unsqueeze(-1)* self.weights
        return pos_encoding(embedings,self.dmodel)

class QueriesGenerator:
    def __init__(self, num_queries, d_model):
        self.num_queries = num_queries
        self.d_model = d_model
        self.queries = torch.randn(num_queries, d_model, requires_grad=True)
    def generate_queries(self):
        return self.queries.unsqueeze(0)   

def get_postional_embedding(d_model, position):
    positional = []
    for i in range(d_model):
        if i % 2 == 0:
            value = torch.sin(torch.tensor(position / (10000 ** (i / d_model))))
        else:
            value = torch.cos(torch.tensor(position / (10000 ** ((i - 1) / d_model))))
        positional.append(value)
    return torch.tensor(positional).reshape(1, -1)

def pos_encoding(x,dmodel):
  for i in range(x.shape[1]):
    x[:,i,:]=x[:,i,:]+get_postional_embedding(dmodel,i)
  return x

def image_to_patches(patch_size, image):
    patches = []
    num_patches = image.shape[1] // patch_size
    for i in range(num_patches):
        for j in range(num_patches):
            p = image[:, (i)*patch_size:(i+1)*patch_size, (j)*patch_size:(j+1)*patch_size, :]
            patches.append(p.flatten(1))
    return torch.stack(patches, dim=1)


def add_cls_token(x,cls_token_id):
    cls_token = torch.full_like(x[:, :1], cls_token_id)  # shape (8, 1)
    return torch.cat([cls_token, x], dim=1)