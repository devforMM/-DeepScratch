# 🎨 Generative Adversarial Networks (GANs)

This directory hosts a complete, "from-scratch" implementation of a Generative Adversarial Network. This module serves as the **ultimate stress test** for the framework's Backpropagation Engine, proving it can handle:
1.  **Non-Stationary Objectives:** The loss landscape changes constantly as two networks compete.
2.  **Complex Computational Graphs:** Managing gradients where one network's output is the other's input.
3.  **Alternating Optimization:** Coordinating two separate optimizers (`Adam`) in a single training loop.

---

## 📐 Theoretical Architecture

The system consists of two distinct neural networks involved in a **Minimax Game**:

### **1. The Generator ($G$)**
* **Input:** Random Latent Vector $z \sim \mathcal{N}(0, 1)$.
* **Function:** $G(z; \theta_g) \to x_{fake}$.
* **Objective:** Map the latent noise to the data space to fool the discriminator.
* **Architecture:** Fully Connected MLP (`Layer` class) with Xavier Normal initialization.

### **2. The Discriminator ($D$)**
* **Input:** Data samples $x$ (either Real or Fake).
* **Function:** $D(x; \theta_d) \to [0, 1]$.
* **Objective:** Output high probability ($1$) for real data and low probability ($0$) for generated data.
* **Architecture:** Fully Connected MLP ending in a Sigmoid activation.

---

## ⚙️ Implementation: Under the Hood

Unlike high-level frameworks where `loss.backward()` handles everything, this implementation manually manages the adversarial flow.

### **The Loss Functions (`GAN.py`)**
We implement the standard Log-Loss with numeric stability ($\epsilon = 1e-8$):

**Discriminator Loss:**
$$J(D) = - \frac{1}{m} \sum_{i=1}^{m} \left[ \log D(x^{(i)}) + \log (1 - D(G(z^{(i)}))) \right]$$

**Generator Loss:**
$$J(G) = - \frac{1}{m} \sum_{i=1}^{m} \log D(G(z^{(i)}))$$

### **The Training Loop (`GAN.ipynb`)**
The `minibatch_SGD_train` function explicitly handles the two-step optimization process per batch:

1.  **Step A: Train Discriminator**
    * Forward Real Data $\to$ Calculate Loss on Real.
    * Forward Noise $\to$ Generate Fake Data $\to$ **Detach Gradients** $\to$ Calculate Loss on Fake.
    * *Backprop:* Update $\theta_d$ to maximize classification accuracy.
2.  **Step B: Train Generator**
    * Forward Noise $\to$ Generate Fake Data.
    * Pass through Discriminator (keep gradients flowing).
    * *Backprop:* Update $\theta_g$ to maximize Discriminator's error.

---

## 🧪 Experiment: 2D Gaussian Distribution Matching

To validate the math without the computational overhead of image rendering, the GAN is tasked with learning a geometric transformation in 2D space.

* **Real Data:** $x = Az + b$ (A Gaussian distribution stretched and shifted).
* **Noise:** Standard Normal distribution.
* **Optimizers:** Custom `Adam` implementation (`lr=0.01`).

### **📊 Convergence Analysis**

The training logs demonstrate a stable **Nash Equilibrium** approach.

| Epoch | Generator Loss | Discriminator Loss | Analysis |
| :--- | :--- | :--- | :--- |
| **1** | `0.7499` | `1.1856` | **Initial State:** $G$ produces random noise. $D$ is confused. |
| **5** | `0.7136` | `1.1298` | **Learning Phase:** $G$ begins to shape the noise to match the data manifold. |
| **10** | **`0.6948`** | **`1.1015`** | **Convergence:** $G$ loss drops below 0.7 (approx $\ln 2$), implying $D$ is guessing with probability ~0.5. The Generator has successfully mimicked the distribution. |

**Output Snippet:**
```text
1 | Gen Loss: 0.7499 | Desc Loss: 1.1856
...
6 | Gen Loss: 0.7084 | Desc Loss: 1.1215
...
10| Gen Loss: 0.6948 | Desc Loss: 1.1015




✅ Key Takeaway
This module proves that the DeepScratch Framework is not limited to simple regression or classification. It successfully computes and propagates gradients through multiple connected networks in a dynamic adversarial setting.
