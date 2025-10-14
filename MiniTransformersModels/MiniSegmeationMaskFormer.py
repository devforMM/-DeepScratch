import sys
sys.path.append("../")
from core.model_structure import Deep_learning_Model
from utils.activations import *
from core.MLp_layer import *
from CNN.Vectorised_Cnn_operations import *
from CNN.Vectorised_Cnn_operations.Vec_cnn_Layers import *
from CNN.Loop_based_cnn.Cnn_layers import *
from Custom_transformers.transformeroperations import *
xavier=XavierNormal()
class CLassesFFN(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,nclasses):
        super().__init__(optimizer, loss)
        self.dmodel=dmodel
        self.nclasses=nclasses
        self.add_layers(
            [Layer(self.dmodel,self.nclasses,"HeNormal")]
        )

class mask_Generator(Deep_learning_Model):
    def __init__(self, optimizer, loss,dmodel,nmasks):
        super().__init__(optimizer, loss)
        self.dmodel=dmodel
        self.nmasks=nmasks
        self.add_layers(
            [Layer(self.dmodel,self.nmasks,"HeNormal")]
        )

            
class VitELayer:
    def __init__(self,vocab_size,dmodel):
        self.vocab_size=vocab_size
        self.dmodel=dmodel
        self.embedigns=xavier.initialize(vocab_size,dmodel)
    def get_postional_embedings(self,x):
        embedings=x.float()@self.embedigns
        return pos_encoding(embedings,self.dmodel)
    

Backbone = Deep_learning_Model(
    "adam", "Crossentropy"
)

Backbone.add_layers([
    vec_Conv2D_layer(3, 32, (3, 3), stride=2, padding=1, initializer="HeNormal"),
    Batch_norm_layer(32),
    Relu6LAyer(),
    vec_Conv2D_layer(32, 32, (3, 3), stride=2, padding=1, initializer="HeNormal"),
    Batch_norm_layer(32),
    Relu6LAyer(),  # ✅
    vec_Conv2D_layer(32, 32, (1, 1), stride=1, padding=0, initializer="HeNormal"),
    Batch_norm_layer(32),
    Relu6LAyer(), # ✅
])

class after_pixel_decoder:
    def __init__(self, dmodel,nmasks):
        super().__init__()
        self.w1 = xavier.initialize( dmodel, nmasks)
    def forward(self, pixels_embedings):
        res=pixels_embedings@self.w1
        return res.permute(0,2,1)
    
class  MLpsegmeteion(Deep_learning_Model):
    def __init__(self, optimizer, loss,d_model):
        super().__init__(optimizer, loss)
        self.d_model=d_model
        self.PrimaryModel=Deep_learning_Model(self.optimizer,self.loss)
        self.PrimaryModel.add_layers(
            [Layer(self.d_model,self.d_model,"HeNormal","relu"),
            Layer(self.d_model,self.d_model,"HeNormal","relu"),
            ]
        )        
    def getMLP_outputs(self,x):
        mlp_outputs=self.PrimaryModel.forward_propagation(x)
        return mlp_outputs
    

from Custom_transformers.Encoder_Decoders import *
import core.metrics as M
class SegModel(Deep_learning_Model):
    def __init__(self, optimizer, loss, dmodel, Nqueries, nclasses, nmasks,backbone_channels):
        super().__init__(optimizer, loss)
        
        # --- hyperparams ---
        self.nmasks = nmasks
        self.dmodel = dmodel
        
        # === Modules ===
        self.backbone = Backbone
        self.image_embedding_layer1 = VitELayer(backbone_channels, dmodel)
        self.image_embedding_layer2 = VitELayer(backbone_channels, dmodel)
        self.after_pixel_decoder = after_pixel_decoder(dmodel, nmasks)
        self.pixel_decoder = DecoderOnly(1, dmodel)
        self.transformer_decoder = Decoder(1, dmodel)
        
        self.mlp_head = MLpsegmeteion(optimizer, loss, dmodel)
        self.query_generator = QueriesGenerator(Nqueries, dmodel)
        
        self.mask_head = mask_Generator(self.optimizer, self.loss, self.dmodel, self.nmasks)
        self.class_head = CLassesFFN(self.optimizer, self.loss, self.dmodel, nclasses)
        
        # === Collect parameters ===
        self.weights.append(self.after_pixel_decoder.w1)
        self.weights.extend(self.backbone.weights)
        self.weights.extend(self.pixel_decoder.weights)
        self.weights.append(self.query_generator.queries)
        self.weights.append(self.image_embedding_layer1.embedigns)
        self.weights.append(self.image_embedding_layer2.embedigns)
        self.weights.extend(self.mask_head.weights)
        self.weights.extend(self.class_head.weights)
    


    # --- fusion mask <-> pixel embeddings ---
    def merge_masks_embeddings_pixels(self, mask_embeddings, pixel_embeddings):
        # mask_embeddings: [B, C, N], pixel_embeddings: [B, C, H*W]
        merged = mask_embeddings.permute(0, 2, 1) @ pixel_embeddings
        return merged

    # --- fusion classes <-> masks ---
    def merge_class_masks(self, class_preds, mask_preds):
        merged = class_preds.permute(0, 2, 1) @ mask_preds
        return merged

    # === Forward propagation ===
    def forward_propagation(self, images):
        #print("\n--- FORWARD START ---")

        # 1. Backbone : extract image features
        image_features = self.backbone.forward_propagation(images)
        #print("image_features:", image_features.shape)

        flat_features = image_features.flatten(2).permute(0, 2, 1)
        #print("flat_features:", flat_features.shape)
        
        # 2. Pixel-level module
        pixel_embeddings = self.image_embedding_layer1.get_postional_embedings(flat_features)
        #print("pixel_embeddings:", pixel_embeddings.shape)

        decoded_pixels = self.pixel_decoder.decode(pixel_embeddings)
        #print("decoded_pixels:", decoded_pixels.shape)

        masks_pixels = self.after_pixel_decoder.forward(decoded_pixels)
        #print("masks_pixels:", masks_pixels.shape)
        
        # 3. Transformer module
        queries = self.query_generator.generate_queries()
        #print("queries:", queries.shape)

        image_embeddings_2 = self.image_embedding_layer2.get_postional_embedings(flat_features)
        #print("image_embeddings_2:", image_embeddings_2.shape)

        transformer_outputs = self.transformer_decoder.decode(image_embeddings_2, queries)
        #print("transformer_outputs:", transformer_outputs.shape)
        
        # 4. MLP head for query features
        mlp_outputs = self.mlp_head.forward_propagation(transformer_outputs)
        #print("mlp_outputs:", mlp_outputs.shape)
        
        # 5. Predict classes and masks
        class_predictions = self.class_head.forward_propagation(mlp_outputs)
        #print("class_predictions:", class_predictions.shape)

        mask_embeddings = self.mask_head.forward_propagation(mlp_outputs).permute(0,2,1)
        #print("mask_embeddings:", mask_embeddings.shape)
        
        # 6. Merge mask embeddings with pixel embeddings
        mask_predictions = self.merge_masks_embeddings_pixels(mask_embeddings, masks_pixels)
        #print("mask_predictions:", mask_predictions.shape)
        
        # 7. Combine class predictions with mask predictions
        final_output = self.merge_class_masks(class_predictions, mask_predictions)
        #print("final_output:", final_output.shape)

        #print("--- FORWARD END ---\n")
        return final_output
    

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
