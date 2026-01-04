👁️ CNN Module: Handcrafted Vision Engine (Manual Logic)

<img width="387" height="130" alt="image" src="https://github.com/user-attachments/assets/066a8894-2ea9-4ef8-b871-a8da999f2ebc" 

This module represents a deep dive into Computer Vision architectures. While it leverages PyTorch's .backward() for gradient computation, every single layer, operation, and architectural decision (like ResNet) has been built from scratch. It is a bridge between manual mathematical control and modern optimization.

🏗️ The Hybrid Approach
The philosophy here is "Logical Transparency". By implementing the layers manually, I maintain full control over the feature extraction process while ensuring numerical stability through PyTorch's autograd engine.

📁 Loop_based_cnn/ (The Pedagogical Engine)
Built for those who want to see how data moves through kernels without the "magic" of high-level APIs.

Manual Kernel Sliding: Explicit implementation of how a filter convolved over an image.

Structure over Abstraction: Defines Cnn_layers.py to show the raw relationship between inputs, weights, and biases.

Initializers: Includes Cnn_initializers.py to control how neurons start their learning journey (He/Xavier).

📁 Vectorised_Cnn_operations/ (The Performance Engine)
This is where theory meets speed.

GEMM Optimization: Convolutions are transformed into Matrix Multiplications to leverage modern CPU/GPU efficiency.

Seamless Integration: Designed to work within a training loop that utilizes PyTorch tensors, allowing for high-speed training on datasets like MNIST or CIFAR.

🔬 Key Components
Handcrafted Layers:

Custom Conv2D: Precise control over padding, stride, and channel mapping.

Pooling: Manual implementation of spatial reduction logic (Max/Average).

Batch Normalization: Custom batch_normalization_Layer.py to handle internal covariate shift.

Architectural Mastery:

ResNet (resnet.py): A from-scratch implementation of the Residual Network. It proves that the framework can handle complex "Skip Connections" and deep hierarchies.


/>
