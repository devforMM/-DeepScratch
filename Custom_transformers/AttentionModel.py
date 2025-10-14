import torch
from transformeroperations import *
from Custom_transformers.Encoder_Decoders import *
import sys
sys.path.append("../")
import core.metrics as  M
from core.model_structure import Deep_learning_Model
class AttentionBasedModel(Deep_learning_Model):
    def __init__(self, optimizer, loss,vocab_size,dmodel,modelType):
        super().__init__(optimizer, loss)
        self.ModelType=modelType
        self.enocders=[]
        self.decoders=[]
        self.weights=[]
        self.layers=[]
        self.EmbedingLayer=NormalAndPostionalEmbeding_layer(vocab_size,dmodel)
        self.weights.extend(self.EmbedingLayer.emebding_layer.embeddings)
    def AddEncoder(self,num_heads,dmodel):
        E=Encoder(num_heads,dmodel)
        self.enocders.append(E)
        self.weights.extend(E.weights)
    def AddDecoder(self,num_heads,dmodel):
        D=Decoder(num_heads,dmodel)
        self.decoders.append(D)
        self.weights.extend(D.weights)
    
    def AddDecoderOnly(self,num_heads,dmodel):
        D=DecoderOnly(num_heads,dmodel)
        self.decoders.append(D)
        self.weights.extend(D.weights)
        
    def generate_embedings(self,x):
        embedings=self.EmbedingLayer.get_final_embedings(x)
        return embedings
        
    def Encode(self,embedings):
        for encoder in self.enocders:
            embedings=encoder.en(embedings)
        return embedings
    
    def decode(self,encoder_scores,target_sequence):
        for d in self.decoders:
            if isinstance(d,Decoder):
             target_sequence=d.decode(encoder_scores,target_sequence)
            elif isinstance(d,DecoderOnly):
                target_sequence=d.decode(target_sequence)
        return target_sequence
    def forward_propagation(self, x,target_sequence=None):
        """
        Perform a forward pass through all layers.
        """
        input_emebedings=self.EmbedingLayer.get_final_embedings(x)
        if self.ModelType=="EncoderDecoder":
            target_embedings=self.EmbedingLayer.get_final_embedings(target_sequence)
            encoder_results=self.Encode(input_emebedings)
            decoder_results=self.decode(encoder_results,target_embedings)
            mlp_inputs=decoder_results
        elif self.ModelType=="EncoderOnly":
            encoder_results=self.Encode(input_emebedings)
            mlp_inputs=encoder_results
        elif self.ModelType=="DecoderOnly":
            target_embedings=self.EmbedingLayer.get_final_embedings(target_sequence)            
            decoder_results=self.decode(target_embedings)
            mlp_inputs=decoder_results
        for l in self.layers:
            mlp_inputs=l.forward(mlp_inputs)  
        logits=mlp_inputs
        return logits      
                        
    def minibatch_SGD_train(self, epochs, x_train, y_train, x_val, y_val, batch_size, learning_rate,train_target=None,validation_target=None, accuracy=False, early_stopping=False, patience=None,weight_decay=False):
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
                
                train_pred=self.forward_propagation(x_batch,train_target)
                
                
                loss = self.loss.compute_loss(train_pred, y_batch)
                loss.backward()

                epoch_loss += loss.item()

                if accuracy:
                    softmax_train_scores = self.loss.softmax(train_pred)
                    acc = M.accuracy(y_batch, softmax_train_scores)
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
                val_pred = self.forward_propagation(x_val,validation_target)
                val_loss = self.loss.compute_loss(val_pred, y_val).item()
                val_losses.append(val_loss)

                if accuracy:
                    softmax_val_scores = self.loss.softmax(val_pred)
                    val_acc = M.accuracy(y_val, softmax_val_scores)
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
