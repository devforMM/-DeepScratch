# 🔄 MiniRNN: From-Scratch Sequence Models

**A pure PyTorch implementation of Recurrent Architectures (RNN, LSTM,
GRU) built from scratch without using `torch.nn` modules.**

This module explores sequence modeling by manually implementing the
mathematical gates, state updates, and gradient calculations. It
features a custom training loop designed for **Backpropagation Through
Time (BPTT)**.

------------------------------------------------------------------------

## 📂 File Structure

  -------------------------------------------------------------------------
  File                      Class / Function        Description
  ------------------------- ----------------------- -----------------------
  **`Rnn_model.py`**        `Rnn_model`             The main orchestrator.
                                                    Handles the forward
                                                    pass over time steps
                                                    and the **BPTT**
                                                    training loop.

  **`Rnn_Layers.py`**       `Rnn_layer`             Vanilla Recurrent Layer
                                                    (Elman RNN).

                            `LStm`                  Long Short-Term Memory
                                                    (Input, Forget, Output
                                                    gates, Cell State).

                            `Gru`                   Gated Recurrent Unit
                                                    (Reset, Update gates).

                            `EmbeddingLayer`        Learnable lookup table
                                                    for tokens.

  **`rnn_operations.py`**   `tokenize`, `get_xy`    Utilities for text
                                                    processing and tensor
                                                    preparation.
  -------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Architecture & Theory

Unlike standard Feed-Forward networks, these models maintain a **Hidden
State ($h_t$)** that carries information from previous time steps to the
current one.

### 1. Vanilla RNN (`Rnn_layer`)

The simplest form of recurrence.

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$$

### 2. LSTM (`LStm`)

Introduced to solve the Vanishing Gradient problem. It maintains a
**Cell State ($C_t$)** separate from the hidden state, regulated by
three gates.

-   **Forget Gate ($f_t$):** What to remove from history?
-   **Input Gate ($i_t$):** What new information to store?
-   **Output Gate ($o_t$):** What to output to the next layer?

### 3. GRU (`Gru`)

A simplified version of LSTM that merges states.

-   **Update Gate ($z_t$):** Decides how much past information to keep.
-   **Reset Gate ($r_t$):** Decides how much past information to ignore.

------------------------------------------------------------------------

## ⚙️ Implementation "Under the Hood"

The unique aspect of this framework is the manual management of the
temporal computational graph.

### 1. The Time Loop (`train_rnn_sgd`)

``` python
for t in range(x_batch.shape[1]):
    train_pred_t = self.forward_propagation(batchtrain_embedings[:, t, :])
    lt = self.loss.compute_loss(train_pred_t, y_batch[:, t, :])
    batch_losses += lt
```

### 2. Mixed Propagation

``` python
if isinstance(layer, (Rnn_layer, Gru, LStm)):
    y = layer.hidden_state(y)
elif isinstance(layer, Layer):
    y = layer.forward(y)
```

### 3. Truncated BPTT & Memory Management

``` python
if l.ht is not None:
    l.ht = l.ht.detach()
if isinstance(l, LStm) and l.ct is not None:
    l.ct = l.ct.detach()
```

------------------------------------------------------------------------

## 🚀 Usage Example

``` python
import torch
from Rnn.Rnn_model import Rnn_model
from Rnn.Rnn_Layers import LStm, EmbeddingLayer
from core.MLp_layer import Layer

vocab_size = 27
d_model = 64
hidden_dim = 128

model = Rnn_model(optimizer="adam", loss="Crossentropy")

model.add_layers([
    EmbeddingLayer(vocab_size, d_model),
    LStm(d_model, hidden_dim),
    Layer(hidden_dim, vocab_size, "linear")
])

losses = model.train_rnn_sgd(
    epochs=50,
    x_train=x_train, y_train=y_train,
    x_val=x_val, y_val=y_val,
    batch_size=32,
    learning_rate=0.01,
    accuracy=True
)
```

------------------------------------------------------------------------

## 🧪 Benchmarks & Validation

### 1. Proof of Concept: Synthetic Data

**Notebook:** `RnnRandomData.ipynb`

Model: `Gru(64) → Layer(Linear)`\
Result: **90.6% accuracy in 10 epochs**

### 2. Real-World Comparison: English Language Modeling

**Notebooks:** `PytorchRnnEnglish.ipynb` vs `CustomRnnEnglish.ipynb`

  Model   Implementation   Epochs   Train Acc   Val Acc
  ------- ---------------- -------- ----------- ---------
  LSTM    torch.nn         10       87.92%      88.26%
  LSTM    MiniRNN          10       88.37%      86.58%

------------------------------------------------------------------------

## ✅ Conclusion

MiniRNN achieves near-parity with PyTorch's native recurrent layers,
validating the correctness of its manually implemented gates, gradients,
and BPTT logic.

*Slight validation differences are attributed to different weight
initialization strategies.*
