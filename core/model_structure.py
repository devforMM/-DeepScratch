import sys
sys.path.append("../")
import core.optimizers as O
import torch
import core.losses as l
import core.metrics as M
from CNN.Loop_based_cnn.Cnn_layers import *
from core.MLp_layer import Layer,EmbeddingLayer
from  Rnn.Rnn_Layers import *


class Deep_learning_Model:
    def __init__(self, optimizer, loss):
        """
        Initialize the model with the selected optimizer and loss function.
        """
        self.layers = []
        self.weights = []

        # Select optimizer
        if optimizer == "gradient_descent":
            self.optimizer = O.GradientDescent()
        elif optimizer == "momentum":
            self.optimizer = O.Momentum()
        elif optimizer == "adagrad":
            self.optimizer = O.Adagrad()
        elif optimizer == "rmsprop":
            self.optimizer = O.RMSProp()
        elif optimizer == "adam":
            self.optimizer = O.Adam()

        # Select loss function
        if loss == "Mse":
            self.loss = l.MSE()
        elif loss == "Mae":
            self.loss = l.MAE()
        elif loss == "Crossentropy":
            self.loss = l.CrossEntropy()
        elif loss == "BinaryCrossentropy":
            self.loss = l.BinaryCrossEntropy()
    def forward_propagation(self, x):
        """
        Perform a forward pass through all layers.
        """
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward_propagation(self, lr, t):
        """
        Apply gradient updates to all trainable parameters.
        """
        for w in self.weights:
            if isinstance(self.optimizer, O.Adam):
                w.data = self.optimizer.update(w, lr, t)
            else:
                w.data = self.optimizer.update(w, lr)

        for w in self.weights:
            w.grad.zero_()

    def add_layers(self, layers):
        """
        Add layers to the model and track trainable parameters.
        """
        for l in layers:
            self.layers.append(l)
            if isinstance(l, Layer):
                self.weights.append(l.w)
            elif isinstance(l, Conv_layer):
                for k in l.kernels:
                    self.weights.append(k)
            elif isinstance(l, Batch_norm_layer):
                self.weights.append(l.gamma)
                self.weights.append(l.beta)
            elif isinstance(l,Rnn_layer) or isinstance(l,Gru) or isinstance(l,LStm):
                for weight in l.w:
                    self.weights.append(weight)
            elif isinstance(l,EmbeddingLayer):
                 self.weights.append(
                     l.embeddings
                 )
    

    def batch_gd_train(self, epochs, x_train, y_train, x_val, y_val, learning_rate, accuracy=False, early_stopping=False, patience=None):
        """
        Batch Gradient Descent training: one full batch per epoch.
        """
        losses, val_losses = [], []
        accuracies, val_accuracies = [], []
        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            # Forward and loss computation
            train_pred = self.forward_propagation(x_train)
            loss = self.loss.compute_loss(train_pred, y_train)
            loss.backward()

            # Parameter update
            self.backward_propagation(learning_rate, epoch+1)

            train_acc, val_acc = 0.0, 0.0
            if accuracy:
                softmax_train_scores = l.CrossEntropy.softmax(None, train_pred)
                train_acc = M.accuracy(y_train, softmax_train_scores)
                accuracies.append(train_acc)

            # Validation phase
            with torch.no_grad():
                val_pred = self.forward_propagation(x_val)
                val_loss = self.loss.compute_loss(val_pred, y_val).item()

                if accuracy:
                    softmax_val_scores = l.CrossEntropy.softmax(None, val_pred)
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

            losses.append(loss.item())
            val_losses.append(val_loss)

            # Epoch log
            if accuracy:
                print(f"{epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            else:
                print(f"{epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        return (losses, val_losses, accuracies, val_accuracies) if accuracy else (losses, val_losses)

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

                train_pred = self.forward_propagation(x_batch)
                
                loss = self.loss.compute_loss(train_pred, y_batch)
                loss.backward()

                epoch_loss += loss.item()

                if accuracy:
                    softmax_train_scores = l.CrossEntropy.softmax(None, train_pred)
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
                    softmax_val_scores = l.CrossEntropy.softmax(None, val_pred)
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

    def SGD_train(self, epochs, x_train, y_train, x_val, y_val, learning_rate, accuracy=False, early_stopping=False, patience=None):
        """
        Stochastic Gradient Descent: one sample at a time.
        """
        losses, val_losses = [], []
        accuracies, val_accuracies = [], []
        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            indices = torch.randperm(len(x_train))
            x_train, y_train = x_train[indices], y_train[indices]

            epoch_loss, epoch_acc = 0.0, 0.0

            for i in range(len(x_train)):
                xi, yi = x_train[i].unsqueeze(0), y_train[i].unsqueeze(0)

                train_pred = self.forward_propagation(xi)
                loss = self.loss.compute_loss(train_pred, yi)
                loss.backward()

                epoch_loss += loss.item()

                if accuracy:
                    softmax_train_scores = l.CrossEntropy.softmax(None, train_pred)
                    acc = M.accuracy(yi, softmax_train_scores)
                    epoch_acc += acc

            # Update parameters after each epoch
            with torch.no_grad():
                self.backward_propagation(learning_rate, epoch+1)

            epoch_loss /= len(x_train)
            losses.append(epoch_loss)

            if accuracy:
                epoch_acc /= len(x_train)
                accuracies.append(epoch_acc)

            # Validation phase
            with torch.no_grad():
                val_pred = self.forward_propagation(x_val)
                val_loss = self.loss.compute_loss(val_pred, y_val).item()
                val_losses.append(val_loss)

                if accuracy:
                    softmax_val_scores = l.CrossEntropy.softmax(None, val_pred)
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
