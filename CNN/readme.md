👁️ NeuroCore CNN Module: The Hybrid Vision Engine
NeuroCore CNN is a research-grade implementation of Convolutional Neural Networks. It strips away the abstraction of standard deep learning libraries to expose the raw mathematical operations of Computer Vision.

This module is unique because it implements the Forward Pass logic entirely from scratch (using loops or vectorization) while leveraging PyTorch's autograd engine solely for gradient computation (.backward()). This offers the perfect balance between educational transparency and training stability.

📂 Module Architecture
The module is split into two distinct execution engines:

<img width="900" height="336" alt="image" src="https://github.com/user-attachments/assets/acba13fb-b90c-4182-ac3d-1074b8bb67a5" />
🐢 Engine 1: Loop-Based (Educational)
Located in Loop_based_cnn/

This engine implements convolutions in their most raw algorithmic form. It is designed to demonstrate exactly how a kernel slides over an input volume.

1. Atomic Convolution (Conv2D)
Unlike standard frameworks that hide the sliding window, this implementation uses explicit Python loops to calculate the dot product at every spatial location.

Input: 3D Tensor (Channels, Height, Width)

Kernel: 3D Tensor (Channels, kH, kW)

Output: 2D Map (H_out, W_out) (Single Feature Map)

<img width="874" height="415" alt="image" src="https://github.com/user-attachments/assets/3fb7722f-4b31-4541-8cc3-4ff916186463" />

2. Manual Pooling
Implements Max Pooling by manually slicing the tensor windows and selecting the maximum value.


🐇 Engine 2: Vectorized (Performance)Located in Vectorised_Cnn_operations/This engine allows for training on real datasets (MNIST, CIFAR) by replacing loops with PyTorch tensor operations.1. 4D Kernel SupportUnlike the loop-based engine which handles single filters, the vectorized layers initialize full 4D weight tensors to handle entire batches of data simultaneously.Weight Shape: (Output_Channels, Input_Channels, Kernel_H, Kernel_W)2. Optimized InitializersCustom classes in Vec_cnn_Layers.py implement advanced initialization strategies to ensure gradient stability:Xavier (Glorot): Ideal for Sigmoid/Tanh activations.Normal: 
<img width="657" height="290" alt="image" src="https://github.com/user-attachments/assets/c5fc13e6-0983-4c40-885c-4622c251b20c" />

🧠 The Architecture: ResNetLocated in resnet.pyThe framework includes a fully functional implementation of Residual Networks (ResNet). It utilizes the custom components defined above to build:Residual Blocks: With manual skip connections ($F(x) + x$).Bottleneck Layers: For deeper variants (ResNet50+).Global Average Pooling: Implemented manually before the final Dense layer.

🎯 Key Takeaways
Transparency: You see exactly how padding changes dimensions and how stride skips pixels.

Mathematics: Initializers are coded from their statistical formulas, not just imported strings.

Flexibility: The system is modular. You can mix a Loop-based layer (for debugging) with a Vectorized layer (for speed).

📜 License
MIT License. Built strictly for educational and research purposes.
