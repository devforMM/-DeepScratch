import sys
import torch
import random
from PIL import Image, ImageDraw
sys.path.append("../")
from core.model_structure import Deep_learning_Model
from utils.activations import *
from CNN.Vectorised_Cnn_operations import *
from CNN.Vectorised_Cnn_operations.Vec_cnn_Layers import *
from CNN.Loop_based_cnn.Cnn_layers import *
from CNN.resnet import Resnet50


from core.model_structure import *
from Custom_transformers.transformeroperations import *
from Custom_transformers.Encoder_Decoders import *
from core.MLp_initializers import *
class VitELayer:
    def __init__(self,vocab_size,dmodel):
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.embedigns=xavier.initialize(vocab_size,dmodel)
    def get_postional_embedings(self,x):
        embedings=x.float()@self.embedigns
        return pos_encoding(embedings,self.dmodel)
    

class CLassesFFN(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,nclasses):
        super().__init__(optimizer, loss)
        self.dmodel=dmodel
        self.nclasses=nclasses
        self.add_layers(
            [Layer(self.dmodel,self.nclasses,"HeNormal")]
        )



class BoxesFFN(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel):
        super().__init__(optimizer, loss)
        self.dmodel=dmodel
        self.add_layers(
            [Layer(self.dmodel,self.dmodel,"HeNormal","relu"),
            Layer(self.dmodel,self.dmodel,"HeNormal","relu"),
            Layer(self.dmodel,self.dmodel,"HeNormal","relu"),
            Layer(self.dmodel,self.dmodel,"HeNormal","relu"),
            Layer(self.dmodel,4,"HeNormal","sigmoid")]
        )

        
class DetrModel(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,Nqueries,nclasses,numchanels):
            super().__init__(optimizer, loss)
            self.dmodel=dmodel
            self.CNN=Resnet50
            self.nclasses=nclasses
            self.BoxesFFnModel=BoxesFFN(self.optimizer,"Mse",dmodel)
            self.ClassesMLpModel=CLassesFFN(self.optimizer,self.loss,dmodel,nclasses)
            self.QueryEGenerator=QueriesGenerator(Nqueries,dmodel)
            self.encoder=Encoder(1,dmodel)
            self.decoder=Decoder(1,dmodel)
            self.embeding_layer=VitELayer(numchanels,dmodel)
            self.weights.append(self.embeding_layer.embedigns)
            self.weights.extend(self.BoxesFFnModel.weights)
            self.weights.extend(self.ClassesMLpModel.weights)
            self.weights.append(self.QueryEGenerator.queries)
            self.weights.extend(self.encoder.weights)
            self.weights.extend(self.decoder.weights)

    def forward_propagation(self, image):
        # Génération des queries
        queries = self.QueryEGenerator.generate_queries()
        #print("Shape queries :", queries.shape)

        # Passage dans le CNN
        image_features = self.CNN.forward_propagation(image)
        #print("Shape image_features :", image_features.shape)

        # Flatten + permutation
        flat_image_features = image_features.flatten(2).permute(0, 2, 1)
        #print("Shape flat_image_features :", flat_image_features.shape)

        # Ajout des embeddings positionnels
        image_features_embedings = self.embeding_layer.get_postional_embedings(flat_image_features)
        #print("Shape image_features_embedings :", image_features_embedings.shape)

        # Encodage par l'encoder
        encoded_image_features = self.encoder.encode(image_features_embedings)
        #print("Shape encoded_image_features :", encoded_image_features.shape)

        # Décodage
        decoder_results = self.decoder.decode(encoded_image_features, queries)
        #print("Shape decoder_results :", decoder_results.shape)

        # Passage dans le FFN pour les boxes
        boxes_socres = self.BoxesFFnModel.forward_propagation(decoder_results)
        #print("Shape boxes_socres :", boxes_socres.shape)

        # Passage dans le MLP pour les classes
        classes_scores = self.ClassesMLpModel.forward_propagation(decoder_results)
        #print("Shape classes_scores :", classes_scores.shape)

        return boxes_socres, classes_scores


    def minibatch_SGD_train(self, epochs, x_train, y_train, x_val, y_val, batch_size, learning_rate,
                            accuracy=False, early_stopping=False, patience=None, weight_decay=False):
        """
        Mini-Batch SGD training for DETR-like model (boxes + classes)
        """
        true_classes_train, true_boxes_train = y_train
        true_classes_val, true_boxes_val = y_val


        true_classes_train=torch.eye(self.nclasses)[true_classes_train]
        true_classes_val=torch.eye(self.nclasses)[true_classes_val]
            
        
        losses, val_losses = [], []
        accuracies, val_accuracies = [], []
        num_batches = len(x_train) // batch_size
        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            indices = torch.randperm(len(x_train))
            x_train = x_train[indices]
            true_boxes_train = true_boxes_train[indices]
            true_classes_train = true_classes_train[indices]

            epoch_loss, epoch_acc = 0.0, 0.0


            for i in range(num_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                x_batch = x_train[start:end]
                boxes_batch = true_boxes_train[start:end]
                classes_batch = true_classes_train[start:end]

                # --- Forward ---
                pred_boxes, pred_classes = self.forward_propagation(x_batch)
    

                # --- Loss ---

                boxes_loss = self.BoxesFFnModel.loss.compute_loss(pred_boxes, boxes_batch)
                classes_loss = self.loss.compute_loss(pred_classes, classes_batch)
                loss = (boxes_loss + classes_loss) / 2

                # --- Backward ---
                loss.backward()
                self.backward_propagation(learning_rate, epoch+1)

                epoch_loss += loss.item()

                # --- Accuracy ---
                if accuracy:
                    
                    softmax_scores = self.loss.softmax(pred_classes)

                    acc = M.accuracy(classes_batch, softmax_scores.squeeze(1))
                    epoch_acc += acc

            # === Epoch end ===
            epoch_loss /= num_batches
            losses.append(epoch_loss)

            if accuracy:
                epoch_acc /= num_batches
                accuracies.append(epoch_acc)

            # === Validation ===
            with torch.no_grad():
                val_pred_boxes, val_pred_classes = self.forward_propagation(x_val)
                val_boxes_loss = self.BoxesFFnModel.loss.compute_loss(val_pred_boxes, true_boxes_val)
                val_classes_loss = self.loss.compute_loss(val_pred_classes, true_classes_val)
                val_loss = (val_boxes_loss + val_classes_loss) / 2
                val_losses.append(val_loss.item())

                if accuracy:
                    softmax_val = self.loss.softmax(val_pred_classes)
                    val_acc = M.accuracy(true_classes_val, softmax_val.squeeze(1))
                    val_accuracies.append(val_acc)

            # === Early stopping ===
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # === Logs ===
            if accuracy:
                print(f"{epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss.item():.4f} | "
                    f"Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            else:
                print(f"{epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss.item():.4f}")

        return (losses, val_losses, accuracies, val_accuracies) if accuracy else (losses, val_losses)

            
            