import torch
import sys
sys.path.append("../")
from core.model_structure import *
from Custom_transformers.transformeroperations import *
from Custom_transformers.Encoder_Decoders import *

class VitELayer:
    def __init__(self,vocab_size,dmodel):
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.embedigns=torch.randn(vocab_size,dmodel,requires_grad=True)
    def get_postional_embedings(self,x):
        embedings=x.float()@self.embedigns
        return pos_encoding(embedings,self.dmodel)


class MlpClassifier(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,classes):
        super().__init__(optimizer, loss,)
        self.add_layers(
            [Layer(dmodel,128,"HeNormal","relu"),
             Layer(128,classes,"HeNormal"),
             ]
        )
    
class ClassificationVit(Deep_learning_Model):
    def __init__(self, optimizer, loss,classes,dmodel,patch_size,vocab_size):
        super().__init__(optimizer, loss)
        self.patch_size=patch_size
        self.dmodel=dmodel
        self.classes=classes
        self.vocab_size=vocab_size
        self.classifier=MlpClassifier(self.optimizer,self.loss,self.dmodel,self.classes)
        self.EmbedingLayer=VitELayer(48,dmodel)
        self.encoder=Encoder(4,dmodel)
        self.weights.append(self.EmbedingLayer.embedigns)
        self.weights.extend(self.encoder.weights)
        self.weights.extend(self.classifier.weights)
    



    def forward_propagation(self, x):


        # Étape 1 : Diviser l’image en patches
        patches_vector = image_to_patches(self.patch_size, x)

        # Étape 2 : Ajouter le token [CLS]
        final_vector = add_cls_token(patches_vector, self.vocab_size)

        # Étape 3 : Embeddings + Positionnels
        input_embeddings = self.EmbedingLayer.get_postional_embedings(final_vector)

        # Étape 4 : Passage dans l’encodeur Transformer
        encoder_results = self.encoder.encode(input_embeddings)

        # Étape 5 : On récupère le token [CLS] pour la classification
        cls_token = encoder_results[:, 0, :]

        # Étape 6 : Passage dans la tête de classification
        classification_results = self.classifier.forward_propagation(cls_token)

        return classification_results


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
                val_pred = self.forward_propagation(x_val)
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
        