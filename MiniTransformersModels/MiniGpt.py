import torch
import sys
sys.path.append("../")
from core.model_structure import *
from Custom_transformers.transformeroperations import *
from Custom_transformers.Encoder_Decoders import *

class MlpClassifier(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,classes):
        super().__init__(optimizer, loss,)
        self.add_layers(
            [Layer(dmodel,128,"HeNormal","relu"),
             Layer(128,classes,"HeNormal",),
             ]
        )

class MiniGPt(Deep_learning_Model):
    def __init__(self, optimizer, loss,vocab_size,dmodel):
        super().__init__(optimizer, loss)
        self.Mlp=MlpClassifier(self.optimizer,self.loss,dmodel,vocab_size)
        self.gptdecoder=DecoderOnly(4,dmodel)
        self.EmbedingLayer=ELayer(vocab_size,dmodel)
        self.weights.append(self.EmbedingLayer.embedigns)
        self.weights.extend(self.gptdecoder.weights)
        self.weights.extend(self.Mlp.weights)
    
    
    def forward_propagation(self, x):


        # Étape 1 : Positionnal Embeddings
        input_postional_emebedings = self.EmbedingLayer.get_postional_embedings(x)

        # Étape 2 : GPT Decoder
        encoder_results = self.gptdecoder.decode(input_postional_emebedings)

        # Étape 3 : MLP
        mlp_inputs = self.Mlp.forward_propagation(encoder_results)

        return mlp_inputs

    
    def accuracy(self,ypreds,ytrue):
        return (ypreds.argmax(1)==ytrue.argmax(1)).float().mean()*100


    def minibatch_SGD_train(self, epochs, x_train, y_train, x_val, y_val, batch_size, learning_rate, accuracy=False, early_stopping=False, patience=None,weight_decay=False):
        """
        Mini-Batch Stochastic Gradient Descent training.
        """
        losses, val_losses = [], []
        accuracies, val_accuracies = [], []
        num_batches = len(x_train) // batch_size
        best_loss = float('inf')
        counter = 0
        
        
        for epoch in range(epochs):
            indices = torch.randperm(len(x_train))
            x_train, y_train = x_train[indices], y_train[indices]

            epoch_loss, epoch_acc = 0.0, 0.0

            for i in range(num_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                
                train_pred=self.forward_propagation(x_batch)
  
                loss = self.loss.compute_loss(train_pred, y_batch)
                loss.backward()

                epoch_loss += loss.item()

                if accuracy:
                    softmax_train_scores = self.loss.softmax(train_pred)
                    self.y_trueu=y_batch
                    self.ypred=softmax_train_scores
                    acc = self.accuracy(y_batch, softmax_train_scores)
                    epoch_acc += acc

            # Update parameters after each epoch
            with torch.no_grad():
                self.backward_propagation(learning_rate, epoch+1)

            epoch_loss /= num_batches
            losses.append(epoch_loss)

            if accuracy:
                epoch_acc /= num_batches
                accuracies.append(epoch_acc)

            # Validation phase
            with torch.no_grad():
                val_pred = self.forward_propagation(x_val)
                val_loss = self.loss.compute_loss(val_pred, y_val).item()
                val_losses.append(val_loss)

                if accuracy:
                    softmax_val_scores = self.loss.softmax(val_pred)
                    val_acc = self.accuracy(y_val, softmax_val_scores)
                    val_accuracies.append(val_acc)

            # Early stopping check
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Epoch log
            if accuracy:
                print(f"{epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            else:
                print(f"{epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        return (losses, val_losses, accuracies, val_accuracies) if accuracy else (losses, val_losses)