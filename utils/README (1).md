# 🛠️ MiniDeep Utilities

**A collection of standalone helper modules for activation functions, normalization, data management, and training control.**

The `utils` module provides the essential building blocks that support the core engine. It isolates specific mathematical operations (like activation formulas) and data processing logic from the main model structure, making the codebase modular and reusable.

---

## 📂 Module Structure

```text
utils/
├── activations.py                # Manual implementations of activation functions
├── batch_normalization_Layer.py  # Custom Batch Normalization logic
├── droupout_Layer.py             # Inverted Dropout implementation
├── data_manipulation.py          # Data splitting and K-Fold Cross-Validation
├── learning_rate.py              # Learning Rate Schedulers (Cosine, Warmup, etc.)
└── weight_decay.py               # L2 Regularization utilities
```

---

## ⚙️ Components Breakdown

### 1. Activation Functions (`activations.py`)

Defines the non-linear transformations applied to layer outputs.

- **ReLU**: Rectified Linear Unit  
  \( f(x) = \max(0, x) \)

- **LeakyReLU**: Allows a small gradient when the unit is not active  
  \( f(x) = \max(\beta x, x) \)

- **Sigmoid**: Squashing function (typically used for binary classification)

- **Tanh**: Hyperbolic Tangent, outputting values between -1 and 1

---

### 2. Normalization & Regularization

#### Batch Normalization Layer (`batch_normalization_Layer.py`)

Implements Batch Normalization to stabilize training by normalizing layer inputs.

**Formula:**  
\[
y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

- **Learnable parameters**: scale (\(\gamma\)) and shift (\(\beta\))  
- **Statistics**: mean (\(\mu\)) and variance (\(\sigma^2\)) computed across the batch

#### Dropout Layer (`droupout_Layer.py`)

Implements **Inverted Dropout**.

- Randomly zeroes out elements of the input tensor with probability \(p\)
- Remaining elements are scaled by \(\frac{1}{1-p}\) during training
- No scaling needed during inference

---

### 3. Data Manipulation (`data_manipulation.py`)

Helper functions to manage datasets without external libraries like Scikit-Learn.

#### `split_data(x, y, test_size)`

- Randomly shuffles the dataset
- Splits tensors into Training and Validation sets  
- Example: `test_size=0.3` → 70% train / 30% validation

#### `cross_validation(k)`

- Implements **K-Fold Cross-Validation**
- Divides the dataset into \(k\) equal folds to evaluate model robustness

---

### 4. Training Control (`learning_rate.py`)

Dynamic learning rate adjustment strategies to improve convergence.

| Scheduler Type | Description | Update Rule |
|---------------|------------|-------------|
| factor | Exponential decay | \( lr_{new} = lr \times 0.9 \) |
| multistep | Step decay | \( lr_{new} = lr / 2 \) |
| cosine | Cosine Annealing | Cosine curve towards `final_lr` |
| warmup | Linear Warmup | Linear increase during warmup steps |

---

## 🚀 Usage Examples

### Learning Rate Schedulers

```python
from utils.learning_rate import learning_rate_scheduler

current_lr = 0.01
for epoch in range(epochs):
    new_lr = learning_rate_scheduler(
        type="cosine",
        lr=current_lr,
        epoch=epoch,
        total_epochs=epochs
    )
    # Update optimizer with new_lr
```

---

### Data Splitting

```python
import torch
from utils.data_manipulation import split_data

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100, 1))

x_train, y_train, x_val, y_val = split_data(X, y, test_size=0.3)

print(f"Train size: {len(x_train)}, Val size: {len(x_val)}")
```

---

### Manual Batch Normalization

```python
import torch
from utils.batch_normalization_Layer import Batch_normalization_layer

bn = Batch_normalization_layer(beta=0.0, gamma=1.0)
input_data = torch.randn(32, 64)

normalized_output = bn.forward(input_data)
```

---

## 📌 Notes

- All modules are implemented **from scratch**
- Designed for **educational purposes** and **full control** over training internals
- No dependency on high-level ML utilities like Scikit-Learn

---

Happy hacking 🚀
