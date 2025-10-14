import torch

class Droupout_layer:
     def __init__(self,p):
          self.p=p
     def add_dropout(self,outputs):
        mask = torch.ones_like(outputs)
        indices = torch.randperm(mask.numel())[:int(mask.numel()*self.p)]
        mask = mask.flatten()
        mask[indices] = 0
        mask = mask.reshape(outputs.shape)
        out_drop = outputs * mask * (1.0 / (1.0 - self.p))
        return out_drop
