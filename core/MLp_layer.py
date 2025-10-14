import torch
from core.MLp_initializers import *

class Layer:
     def __init__(self,input_shape,nbr_neurones,initializer,acitvation=None):
          self.activation=acitvation
          if initializer !=None:
            if initializer=="Xaviernormal":
                init= XavierNormal()
            elif initializer=="HeNormal":
                init=HeNormal()
            elif initializer=="XavierUniform":
                init=XavierUniform()
            elif initializer=="HeUniform":
                init=HeUniform()
            self.w= init.initialize(input_shape,nbr_neurones)
          else:
              self.w=torch.randn(input_shape,nbr_neurones,requires_grad=True)       

     def forward(self,x):
        z = x @ self.w
        if self.activation is None:
            return z
        elif self.activation == "relu":
            return torch.maximum(torch.tensor(0.0), z)
        elif self.activation == "lakyrelu":
            return torch.maximum(torch.tensor(0.001), z)
        elif self.activation == "sigmoid":
            return 1 / (1 + torch.exp(-z))
        elif self.activation == "tanh":
            return torch.tanh(z)

        else:

            raise ValueError(f"Unknown activation function {self.activation}")

class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = torch.randn(vocab_size, d_model, requires_grad=True)
    
    def get_embedings(self, x):
        return self.embeddings[x]
    

