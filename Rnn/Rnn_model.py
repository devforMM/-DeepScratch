# Imports
import sys
sys.path.append("../")
import torch
from Rnn.Rnn_Layers import *
from core.model_structure import Deep_learning_Model
import core.metrics as M
import core.losses as Loss
from utils.data_manipulation import split_data
from core.MLp_layer import Layer
# Rnn_model spécialisation de Deep_learning_Model

# Rnn_model spécialisation de Deep_learning_Model
class Rnn_model(Deep_learning_Model):
    def __init__(self, optimizer, loss):
        super().__init__(optimizer, loss)
    def forward_propagation(self,x):
        y=x
        for layer in self.layers:
         if isinstance(layer,Rnn_layer) or isinstance(layer,Gru) or isinstance(layer,LStm):
            y=layer.hidden_state(y)
         elif  isinstance(layer,Layer):
                y=layer.forward(y)
        return y
    


    def train_rnn_sgd(self, epochs, x_train, y_train, x_val, y_val, batch_size, learning_rate, accuracy=False):
        
        loop_plosses, loop_val_losses = [], []
        loop_accuracies, loop_val_accuracies = [], []
        num_batches = x_train.shape[0] // batch_size
        num_val_batches = x_val.shape[0] // batch_size

        for e in range(epochs):
            epoch_loss = []
            epoch_acc = []

            indices = torch.randperm(len(x_train))

            x_train, y_train = x_train[indices], y_train[indices]

            for i in range(num_batches):
                batch_losses = torch.tensor(0.0)
                batch_accs = 0.0

                start, end = i * batch_size, (i + 1) * batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                
                if isinstance(self.layers[0],EmbeddingLayer):
                 batchtrain_embedings=self.layers[0].get_embedings(x_batch)
                else:
                    batchtrain_embedings=x_batch

   
                for t in range(x_batch.shape[1]):
                    train_pred_t = self.forward_propagation(batchtrain_embedings[:, t, :])
                    lt = self.loss.compute_loss(train_pred_t, y_batch[:, t, :])
                    batch_losses += lt

                    if accuracy:
                        softmax_train_scores = self.loss.softmax(train_pred_t)

                        train_acc = M.accuracy(y_batch[:, t, :], softmax_train_scores)
                        batch_accs += train_acc


                batch_loss = batch_losses / x_batch.shape[1]
                batch_acc = batch_accs / x_batch.shape[1] if accuracy else None

                batch_loss.backward()

                with torch.no_grad():
                    self.backward_propagation(learning_rate, e+1)


                for l in self.layers:
                    if isinstance(l,Rnn_layer) or isinstance(l,Gru) or isinstance(l,LStm):
                     if l.ht is not None:
                        l.ht = l.ht.detach()
                
                


                epoch_loss.append(batch_loss.item())
                if accuracy:
                    epoch_acc.append(batch_acc)

            with torch.no_grad():
                val_losses = []
                val_accuracies = []

                for i in range(num_val_batches):
                    start, end = i * batch_size, (i + 1) * batch_size
                    x_val_batch, y_val_batch = x_val[start:end], y_val[start:end]
                    batch_loss = torch.tensor(0.0)
                    batch_acc = 0.0
                    

                    if isinstance(self.layers[0],EmbeddingLayer):
                     batchval_embedings=self.layers[0].get_embedings(x_val_batch)
                    else:
                     batchval_embedings=x_val_batch

                    for t in range(x_val_batch.shape[1]):
                        val_pred_t = self.forward_propagation(batchval_embedings[:, t, :])
                        val_lt = self.loss.compute_loss(val_pred_t, y_val_batch[:, t, :])
                        batch_loss += val_lt

                        if accuracy:
                            softmax_val_scores = self.loss.softmax(val_pred_t)
                            
                            val_acc = M.accuracy(y_val_batch[:, t, :], softmax_val_scores)
                            batch_acc += val_acc

                    batch_loss /= x_val_batch.shape[1]
                    batch_acc = batch_acc / x_val_batch.shape[1] if accuracy else None

                    val_losses.append(batch_loss.item())
                    if accuracy:
                        val_accuracies.append(batch_acc)

                val_loss = sum(val_losses)
                val_acc = sum(val_accuracies) / num_val_batches if accuracy else None

            # Affichage des résultats epoch
            e_loss = sum(epoch_loss)


            loop_plosses.append(e_loss)
            loop_val_losses.append(val_loss)
            if accuracy:
                e_acc = sum(epoch_acc) / num_batches
                print(f"Epoch {e+1} | Train Loss: {e_loss:.4f} | Train Acc: {e_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                loop_accuracies.append(e_acc)
                loop_val_accuracies.append(val_acc)
                

            else:
                print(f"Epoch {e+1} | Train Loss: {e_loss:.4f} | Val Loss: {val_loss:.4f}")
                

        if accuracy:
            return loop_plosses,loop_val_losses,loop_accuracies,loop_val_accuracies
        return loop_plosses,loop_val_losses


