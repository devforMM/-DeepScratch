
This module defines a customizable deep learning model class built from scratch using PyTorch tensors and autograd.  
It implements different training strategies like **Batch Gradient Descent**, **Mini-Batch SGD**, and **Stochastic Gradient Descent**, with support for popular optimizers and loss functions coded manually.

---

## 📦 Contents

- `Deep_learning_Model` class
- Training methods:
  - `batch_gd_train()`
  - `minibatch_SGD_train()`
  - `SGD_train()`
- Model management methods:
  - `forward_propagation()`
  - `backward_propagation()`
  - `add_layers()`

---

## 📚 Class Overview

### 📌 Initialization

```python
model = Deep_learning_Model(optimizer="adam", loss="Crossentropy")