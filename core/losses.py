import torch


class MAE:
    def compute_loss(self,y_pred, y_true):
        return torch.abs(y_pred - y_true).mean()



class MSE:
    def compute_loss(self,y_pred, y_true):
        return 0.5 * torch.mean((y_pred - y_true) ** 2)



class CrossEntropy:
    def __init__(self):
        ()
    def softmax(self,x):
        x = x - torch.max(x, dim=1, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    
    def compute_loss(self,y_pred, y_true):
        res = self.softmax(y_pred)
        loss = -torch.mean(y_true*torch.log(res+1e-9))
        return loss

class BinaryCrossEntropy:
        def __init__(self):
         ()
        def sigmoid(self,ypred):
           return 1/(1+torch.exp(ypred))
        def compute_loss(self,y_pred, y_true):
           res=self.sigmoid(y_pred)
           loss = -torch.mean(y_true * torch.log(res) + (1-y_true) * torch.log(1-res))
           return loss


