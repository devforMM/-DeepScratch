# 🧠 MiniDeep Core Engine

**A transparent, educational Deep Learning framework built from scratch
using PyTorch tensors.**

The `core` module is the heart of **MiniDeep**. It replaces high-level
abstractions (`torch.nn`, `torch.optim`) with **manual implementations**
of Backpropagation, Optimization Algorithms, Weight Initialization, and
Layer management.\
The goal is to deeply understand the *mathematics and mechanics* behind
Deep Learning.

------------------------------------------------------------------------

## 📂 Module Structure

``` text
core/
├── model_structure.py    # Model container & training loops
├── MLp_layer.py          # Dense (Fully Connected) & Embedding layers
├── Droupout_layer.py     # Custom Inverted Dropout
├── MLp_initializers.py   # Weight initialization (He, Xavier)
├── optimizers.py         # SGD, Momentum, RMSProp, Adam (manual)
├── losses.py             # Loss functions
└── metrics.py            # Accuracy, Precision, Recall, F1
```

------------------------------------------------------------------------

## ⚙️ Components Breakdown

### 1. Model Orchestrator (`model_structure.py`)

The `Deep_learning_Model` class is the central engine.

**Responsibilities** - Stores layers in execution order - Aggregates
trainable parameters - Controls forward & backward passes - Applies
optimizer updates manually

**Training Modes** - `batch_gd_train` -- full batch Gradient Descent -
`minibatch_SGD_train` -- standard mini-batch training - `SGD_train` --
stochastic (sample-wise) training

**Features** - Early Stopping (patience-based) - Validation monitoring -
Optional accuracy tracking

------------------------------------------------------------------------

### 2. Layers

#### Dense Layer (`MLp_layer.py`)

Implements a fully connected layer:

\[ Z = XW + b \]

**Activations Supported** - ReLU - LeakyReLU - Sigmoid - Tanh

**Initialization** - Integrated with `MLp_initializers.py` - Weights
initialized at layer creation

------------------------------------------------------------------------

#### Dropout Layer (`Droupout_layer.py`)

Implements **Inverted Dropout**.

\[ Output = `\frac{Input \times Mask}{1 - p}`{=tex} \]

-   Scaling is applied **during training**
-   No changes required at inference time

------------------------------------------------------------------------

### 3. Weight Initialization (`MLp_initializers.py`)

  ---------------------------------------------------------------------------------------------
  Method      Distribution              Formula                               Best For
  ----------- ------------------------- ------------------------------------- -----------------
  Xavier      Normal                    ( `\sqrt{2 / (n_{in}+n_{out})}`{=tex} Sigmoid / Tanh
  Normal                                )                                     

  He Normal   Normal                    ( `\sqrt{2 / n_{in}}`{=tex} )         ReLU

  Xavier      Uniform                   ( `\sqrt{6 / (n_{in}+n_{out})}`{=tex} Sigmoid / Tanh
  Uniform                               )                                     

  He Uniform  Uniform                   ( `\sqrt{6 / n_{in}}`{=tex} )         ReLU
  ---------------------------------------------------------------------------------------------

------------------------------------------------------------------------

### 4. Optimizers (`optimizers.py`)

Manual re-implementation of `torch.optim`.

**Supported** - Gradient Descent - Momentum - Adagrad - RMSProp - Adam
(with bias correction)

Adam bias correction: \[ `\hat{m}`{=tex}\_t =
`\frac{m_t}{1 - \beta_1^t}`{=tex}, `\quad`{=tex} `\hat{v}`{=tex}\_t =
`\frac{v_t}{1 - \beta_2^t}`{=tex} \]

------------------------------------------------------------------------

### 5. Losses & Metrics

#### Loss Functions (`losses.py`)

-   MSE / MAE (Regression)
-   Binary Cross Entropy
-   Cross Entropy (Softmax + NLL)

#### Metrics (`metrics.py`)

-   Accuracy
-   Precision
-   Recall
-   F1-Score\
    (Computed via manual confusion matrix)

------------------------------------------------------------------------

## 🚀 Usage Example

``` python
import torch
from core.model_structure import Deep_learning_Model
from core.MLp_layer import Layer
from core.Droupout_layer import Droupout_layer

# Dummy Data
x_train = torch.randn(100, 20)
y_train = torch.randint(0, 2, (100, 1)).float()

# Model
model = Deep_learning_Model(
    optimizer="adam",
    loss="BinaryCrossentropy"
)

# Architecture
model.add_layers([
    Layer((20, 64), 64, initializer="HeNormal", acitvation="relu"),
    Droupout_layer(p=0.2),
    Layer((64, 32), 32, initializer="HeNormal", acitvation="relu"),
    Layer((32, 1), 1, initializer="XavierNormal", acitvation="sigmoid")
])

# Training
losses, val_losses, acc, val_acc = model.minibatch_SGD_train(
    epochs=50,
    x_train=x_train, y_train=y_train,
    x_val=x_train, y_val=y_train,
    batch_size=16,
    learning_rate=0.001,
    accuracy=True,
    early_stopping=True,
    patience=5
)
```

------------------------------------------------------------------------

## ⚠️ Requirements

-   Python 3.x
-   PyTorch\
    *(Used only for tensors and autograd -- no `nn.Module`, no `optim`)*

------------------------------------------------------------------------

## 🎯 Philosophy

MiniDeep Core prioritizes: - Mathematical transparency - Educational
clarity - Explicit control of gradients and updates

This is **not** a speed-optimized framework --- it is a *learning
engine*.
