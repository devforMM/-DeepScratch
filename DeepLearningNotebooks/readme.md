# 📂 Benchmarks & Validation Suite

This directory contains the comprehensive testing suite for the **DeepScratch Framework**. It demonstrates that the custom-built layers, optimizers, and vectorized operations perform on par with standard libraries like PyTorch, validated across the entire complexity spectrum: from a single neuron to Deep Convolutional Networks.

---

## 🧱 1. Foundational Level: Single Neuron Unit Tests
**Goal:** Validate the mathematical core (Backpropagation & Optimizers) on the simplest possible units before building deep networks.

### **A. Linear Regression (Slope Discovery)**
* **File:** `single_perceptron.ipynb`
* **Task:** Learn the function $y = 2x$ using a single neuron without activation.
* **Optimizer:** Vanilla Gradient Descent (`lr=1e-6`).
* **Result:** The weight $w$ converged from a random initialization of **-0.29** to **~1.80** smoothly.
* **Convergence Log:**
    * Start: Loss `9283.74`
    * Epoch 50: Loss `809.09`
    * **Epoch 100: Loss `67.08`** (Converged)

### **B. Binary Classification (Threshold Logic)**
* **File:** `single_perceptron_classifiaction.ipynb`
* **Task:** Learn a hard threshold: $y = 1$ if $x \ge 50$, else $0$.
* **Model:** Single Neuron + **Binary CrossEntropy**.
* **Result:** Loss converged successfully from **7.44** to **0.54**.
* **Significance:** Proves the derivative of the Sigmoid/BCE chain rule is implemented correctly.

---

## 📉 2. Intermediate Level: Stress Tests (MLP)
**Goal:** Verify that the **Adam Optimizer** and **Dense Layers** handle high noise and multi-dimensional data.

### **A. Synthetic Regression (High Noise Handling)**
* **File:** `regression_mlp.ipynb`
* **Data:** `sklearn.make_regression` (500 samples, 5 features, **Noise=10.0**).
* **Architecture:** Deep MLP (Input $\to$ 128 $\to$ 64 $\to$ 5 $\to$ Output).
* **Result:** Despite significant noise, the model filtered the data to find the underlying trend.

| Epoch | Train Loss (MSE) | Observation |
| :--- | :--- | :--- |
| 1 | ~7356.26 | Random Initialization |
| 10 | ~5462.01 | Descent begins |
| **100** | **~41.49** | **Optimal Convergence (Residual Noise)** |

### **B. Real-World Regression (California Housing)**
* **File:** `california_housing.ipynb`
* **Task:** Predict house prices (Continuous data).
* **Result:**
    * Initial Loss: ~12,573
    * **Final Loss: ~2.79**
* **Proof:** Proves that the Backpropagation engine correctly handles real-world regression gradients without instability.

### **C. Classical Classification (Iris)**
* **File:** `iris_classification.ipynb`
* **Task:** Multi-class classification (3 classes).
* **Architecture:** MLP (4 Inputs $\to$ 128 $\to$ 64 $\to$ 5 $\to$ 3 Outputs).
* **Result:**
    * **91.67% Train Accuracy**
    * **86.67% Validation Accuracy**
    * Stable loss decrease from 0.38 to 0.07.

---

## 🧪 3. Geometric "Unit Tests" for CNNs
**Files:** `vec_cnn_shapes.ipynb` (Vectorized) vs `simple_CNN_shapes.ipynb` (Loop-based)

To prove the CNN layers correctly extract spatial features (edges vs. corners), I built a custom data generator `generate_mini_mnist` that creates 5x5 images with specific geometric patterns (Vertical/Horizontal lines, Diagonals, Squares).

**Performance Comparison:**

| Implementation | Epochs to Converge | Final Accuracy | Note |
| :--- | :--- | :--- | :--- |
| **Vectorized CNN** | **4 Epochs** | **100%** | Converged almost instantly. Proves `im2col` logic is perfect. |
| **Loop-based CNN** | 10 Epochs | ~80% | Slower, used for verifying the mathematical sliding window logic ($O(N^4)$). |

---

## 🖼️ 4. Computer Vision: MNIST Digit Recognition
**Files:** `mnist_cnn.ipynb` (Production) & `loop_based_mnist.ipynb` (Educational)

The stress test for the Convolution Engine on the classic handwritten digit dataset.

### **A. Vectorized Implementation (The Stress Test)**
* **Architecture:** Deep CNN (Conv $\to$ BatchNorm $\to$ LeakyReLU $\to$ MaxPool).
* **Tech Stack:** Uses custom `vec_Conv2D_layer` which implements efficient matrix operations (im2col).
* **Result:** Reached **88.4% Accuracy** in just 25 epochs.
* **Key Validation:**
    * Correct handling of Padding and Strides.
    * **Batch Normalization** effectively stabilized the deep network.
    * No exploding gradients.

### **B. Loop-Based Implementation**
* **Goal:** To implement the raw formula: $Output(i,j) = \sum \sum Input \times Kernel$.
* **Result:** Reached **100% accuracy** on a validation subset. Demonstrates the raw mathematical understanding behind the optimized code.

---

### 🛠️ How to reproduce
All notebooks use the core framework. Ensure the root directory is in your path to load the modules.

```python
import sys
sys.path.append("../") 

from core.model_structure import Deep_learning_Model
from core.MLp_layer import Layer
from CNN.Vectorised_Cnn_operations.Vec_cnn_Layers import vec_Conv2D_layer

# Example: Simple MLP Construction
model = Deep_learning_Model("adam", "Crossentropy")
model.add_layers([
    Layer(4, 128, "Xaviernormal", "lakyrelu"),
    Layer(128, 64, "Xaviernormal", "lakyrelu"),
    Layer(64, 3, None) 
])
