import sys

sys.path.append("../")
import core.metrics as M
from core.model_structure import Deep_learning_Model
from utils.activations import *
from core.MLp_layer import *
from CNN.Vectorised_Cnn_operations import *
from CNN.Vectorised_Cnn_operations.Vec_cnn_Layers import *
from CNN.Loop_based_cnn.Cnn_layers import *
from  Custom_transformers.transformeroperations import *
from Custom_transformers.Encoder_Decoders import *

class VitELayer:
    def __init__(self,vocab_size,dmodel):
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.embedigns=torch.randn(vocab_size,dmodel,requires_grad=True)
    def get_postional_embedings(self,x):
        embedings=x.float()@self.embedigns
        return pos_encoding(embedings,self.dmodel)

class ClipModel(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,vocab_size,classes,patch_size):
        super().__init__(optimizer, loss)
        self.pathc_size=patch_size
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.classes=classes
        self.TextEmbedingLayer=ELayer(vocab_size,self.dmodel)
        self.ImageEmbedingLayer=VitELayer(768,dmodel)
        self.TextEncoder=Encoder(4,dmodel)
        self.ImageEncoder=VitEncoder(4,dmodel)
        self.weights.append(self.TextEmbedingLayer.embedigns)
        self.weights.append(self.ImageEmbedingLayer.embedigns)
        self.weights.extend(self.TextEncoder.weights)
        self.weights.extend(self.ImageEncoder.weights)
    
    def SimilarityMatrix(self, texte_embeddings, image_embeddings):
        # Pooling: moyenne sur la dimension de séquence
        # Image: (32, 4, 128) -> (32, 128)
        image_pooled = image_embeddings.mean(dim=1)
        # Texte: (32, 12, 128) -> (32, 128)
        texte_pooled = texte_embeddings.mean(dim=1)
        
        # Normalisation L2
        normalized_image = image_pooled / image_pooled.norm(dim=-1, keepdim=True)
        normalized_texte = texte_pooled / texte_pooled.norm(dim=-1, keepdim=True)
        
        # Similarité cosine: (32, 128) @ (128, 32) = (32, 32)
        similarity_matrix = normalized_image @ normalized_texte.T
        
        return similarity_matrix

    
    def forward_propagation(self, textes, images):
        images_patches = image_to_patches(self.pathc_size, images)

        image_embedings = self.ImageEmbedingLayer.get_postional_embedings(images_patches)

        text_emebedings = self.TextEmbedingLayer.get_postional_embedings(textes)

        encoded_image = self.ImageEncoder.encode(image_embedings)

        encoded_texte = self.TextEncoder.encode(text_emebedings)

        similarity = self.SimilarityMatrix(encoded_texte, encoded_image)

        return similarity


    

    def minibatch_SGD_train(self, epochs, x_train, y_train, x_val, y_val, batch_size, learning_rate, accuracy=False, early_stopping=False, patience=None,weight_decay=False):
        """
        Mini-Batch Stochastic Gradient Descent training.
        """
        images=x_train[1]
        textes=x_train[0]
        losses, val_losses = [], []
        accuracies, val_accuracies = [], []
        num_batches = len(x_train[0]) // batch_size
        best_loss = float('inf')
        counter = 0
        
        for epoch in range(epochs):
            indices = torch.randperm(len(textes))
            textes,images, y_train = textes[indices],images[indices], y_train[indices]
             
            epoch_loss, epoch_acc = 0.0, 0.0
 
            for i in range(num_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                x_text_batch,x_img_batch, y_batch = textes[start:end], images[start:end], y_train[start:end]
                train_pred=self.forward_propagation(x_text_batch,x_img_batch)
                loss = self.loss.compute_loss(train_pred, y_batch)
                loss.backward()

                epoch_loss += loss.item()

                if accuracy:
                    softmax_train_scores = self.loss.softmax(train_pred)
                    y_batch_acc=torch.eye(self.classes)[y_batch]
                    acc = M.accuracy(y_batch_acc, softmax_train_scores)
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
                x_val_texte,x_val_img=x_val[0],x_val[1]
                val_pred = self.forward_propagation(x_val_texte,x_val_img)

                val_loss = self.loss.compute_loss(val_pred, y_val).item()
                val_losses.append(val_loss)

                if accuracy:
                    softmax_val_scores = self.loss.softmax(val_pred)
                    y_val_acc=torch.eye(self.classes)[y_val]
                    val_acc = M.accuracy(y_val_acc, softmax_val_scores)
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
 


