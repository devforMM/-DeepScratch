# 🧠 MiniDeep Framework

**MiniDeep** is a lightweight deep learning framework built entirely from scratch using Python.  
It is designed for **learning, experimentation, and full transparency** into how neural networks work internally — without relying on high-level libraries like PyTorch or TensorFlow for model logic.

MiniDeep focuses on clarity over performance, making it ideal for:
- Students 👩‍🎓👨‍🎓
- Researchers who want full control 🔬
- Developers learning deep learning internals ⚙️

---

## 🎯 Project Goals

- Understand deep learning **from first principles**
- Implement core neural network components manually
- Keep the codebase **modular, readable, and hackable**
- Avoid “black-box” abstractions

---

## 📂 Project Structure

```text
MiniDeep/
├── core/                    # Core neural network engine
│   ├── tensor.py            # Custom Tensor structure & operations
│   ├── layer.py             # Base Layer class
│   ├── model.py             # Model container
│   └── loss.py              # Loss functions
│
├── utils/                   # Standalone utilities
│   ├── activations.py
│   ├── batch_normalization_Layer.py
│   ├── droupout_Layer.py
│   ├── data_manipulation.py
│   ├── learning_rate.py
│   └── weight_decay.py
│
├── optimizers/              # Optimization algorithms
│   ├── sgd.py
│   ├── momentum.py
│   └── adam.py
│
├── examples/                # Training examples & demos
│   ├── linear_regression.py
│   └── classification.py
│
├── tests/                   # Unit tests
├── README.md                # Project documentation
└── requirements.txt
```

---

## 🧩 Core Components

### 1. Tensor Engine
- Custom tensor abstraction
- Manual forward & backward propagation
- Gradient tracking

### 2. Layers
- Dense (Fully Connected)
- Dropout
- Batch Normalization

### 3. Activations
- ReLU / LeakyReLU
- Sigmoid
- Tanh
- Softmax

### 4. Loss Functions
- Mean Squared Error (MSE)
- Binary Cross Entropy
- Categorical Cross Entropy

### 5. Optimizers
- SGD
- Momentum
- Adam

### 6. Training Utilities
- Learning rate schedulers
- Weight decay (L2 regularization)
- Data splitting & cross-validation

---

## 🚀 Example Usage

```python
from core.model import Model
from core.layer import Dense
from utils.activations import relu, sigmoid
from optimizers.adam import Adam

model = Model()
model.add(Dense(10, 32, activation=relu))
model.add(Dense(32, 1, activation=sigmoid))

model.compile(
    optimizer=Adam(lr=0.001),
    loss="binary_crossentropy"
)

model.fit(X_train, y_train, epochs=100)
```

---

## 🧪 Educational Focus

MiniDeep is **not** optimized for speed or large-scale production use.

Instead, it prioritizes:
- Readability over performance
- Explicit math over abstractions
- Debuggability and learning

---

## 📌 Requirements

- Python 3.9+
- NumPy (optional, depending on modules)

```bash
pip install -r requirements.txt
```

---

## 🛠️ Roadmap

- [ ] Convolutional layers (CNN)
- [ ] Recurrent layers (RNN / LSTM)
- [ ] Automatic differentiation engine
- [ ] GPU support (educational)

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to:
- Open issues
- Propose improvements
- Add new layers or utilities

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

Built with ❤️ for learning deep learning from scratch.
