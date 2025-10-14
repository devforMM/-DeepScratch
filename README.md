Deep Learning Framework “From Scratch” — From Perceptrons to Transformers, with Zero Black Boxes.

"If you can't implement it, you don't understand it."
— Richard Feynman
🎯 Philosophy

🧠 Understand, Don’t Just Use
True mastery in deep learning comes from building, not just importing.



🏗️ Educational Foundation
An ideal playground for those who want to truly understand how deep learning works under the hood.

🚀 What’s Implemented — From Scratch
🧩 Core Framework

Custom Layer System with manual forward() and backward() passes

Optimizers: SGD, RMSprop, Adam (manual gradient updates)

Initializers: Xavier, He Normal

Loss Functions: CrossEntropy, MSE, Binary CrossEntropy

Activations: ReLU, Sigmoid, Tanh, Softmax

🧮 All mathematical operations (dot products, convolutions, softmax, etc.) are implemented manually using only low-level PyTorch tensor ops like sum, matmul, log, etc. The only autograd features used are requires_grad and .backward() — everything else is 100% handcrafted.

🧠 Architecture Portfolio

MLPs — Multi-Layer Perceptrons

CNNs — Convolutional Neural Networks

Loop-based (educational) and vectorized (optimized) versions

Custom Conv2D, Pooling, BatchNorm layers

RNNs & LSTMs — Recurrent Neural Networks

Transformers — Full architecture

Multi-Head Attention

Positional Encoding

Encoder, Decoder, and Full Transformer models

🧩 Model Zoo — From Scratch Implementations

MiniGPT — Generative Transformer

MiniBERT — Bidirectional Encoder

MiniViT — Vision Transformer

MiniCLIP — Contrastive Language–Image Model

MiniDETR — Detection Transformer

MiniMaskFormer — Segmentation Transformer   



📦 DeepScratch/
│
├── 📁 core/                         # Core engine: manual forward/backward passes, optimizers, and model base
│   ├── MLp_layer.py                 # Dense layers & initialization (manual linear algebra)
│   ├── MLp_initializers.py          # Xavier, He Normal, Uniform
│   ├── optimizers.py                # SGD, RMSprop, Adam (from scratch)
│   ├── losses.py                    # CrossEntropy, MSE, BCE
│   ├── metrics.py                   # Accuracy, Precision, Recall, F1 (custom)
│   ├── model_structure.py           # Base Model class handling training loop logic
│   ├── Droupout_layer.py            # Custom dropout layer
│   └── *.md                         # Theory explanations (educational docs)
│
├── 📁 CNN/                          # Custom Convolutional Neural Networks
│   ├── 📁 Loop_based_cnn/           # Educational, explicit loop implementations
│   │   ├── Cnn_layers.py            # Manual convolution, pooling, batchnorm
│   │   ├── Cnn_operations.py        # Pixel-by-pixel conv & backprop
│   │   └── Cnn_initializers.py      # Kernel initialization logic
│   │
│   ├── 📁 Vectorised_Cnn_operations/ # Optimized vectorized CNN version
│   │   ├── Vec_cnn_Layers.py
│   │   └── Vectorised_Cnn_operations.py
│   │
│   └── resnet.py                    # Custom handcrafted ResNet implementation
│
├── 📁 Custom_transformers/           # Low-level transformer mechanics
│   ├── transformeroperations.py      # Manual multi-head attention, masking, QKV ops
│   └── Encoder_Decoders.py           # Encoder/Decoder architecture logic
│
├── 📁 Rnn/                           # Recurrent neural networks (from scratch)
│   ├── RNN_oprations.py              # Manual matrix-based RNN/LSTM cell ops
│   ├── Rnn_Layers.py                 # Layer abstraction
│   ├── Rnn_model.py                  # Full sequence model
│   ├── datasets/                     # CSV datasets for multilingual translation
│   └── notebooks/                    # Educational comparisons vs PyTorch
│
├── 📁 MiniTransformersModels/        # Ready-to-train models built on custom blocks
│   ├── MiniGpt.py                    # Generative transformer
│   ├── Minibert.py                   # Bidirectional encoder (BERT)
│   ├── MiniVit.py                    # Vision Transformer
│   ├── MiniClip.py                   # Text-Image contrastive model
│   ├── MiniDetr.py                   # Object detection transformer
│   ├── MiniSegmeationMaskFormer.py   # Segmentation transformer
│   └── test.ipynb                    # Validation notebook
│
├── 📁 GANs/                          # Custom generative adversarial networks (planned/under dev)
│
├── 📁 DeepLearningNotebooks/         # Jupyter notebooks for training and demos
│   ├── single_perceptron.ipynb       # Manual perceptron implementation
│   ├── regression_MLP.ipynb          # Linear regression demo
│   ├── Multi_classification_MLP.ipynb
│   ├── loop_based_mnist.ipynb        # CNN from scratch
│   ├── Vec_Cnn_mnist.ipynb           # Vectorized CNN comparison
│   └── California_housing.ipynb      # Tabular regression example
│
├── 📁 TranfomerModeslNotebooks/      # Training notebooks for each Transformer variant
│   ├── MiniGpt_notebook.ipynb
│   ├── MiniBert_notebook.ipynb
│   ├── MiniClip_Notebook.ipynb
│   ├── MiniDetr_notebook.ipynb
│   ├── MiniVitClassifier_notebook.ipynb
│   └── MiniSegTransformer.ipynb
│
├── 📁 utils/                         # Utility layers and helpers
│   ├── activations.py                # ReLU, Sigmoid, Tanh, Softmax (manual)
│   ├── batch_normalization_Layer.py  # Custom batchnorm layer
│   ├── data_manipulation.py          # Mini data loaders & preprocessing tools
│   ├── learning_rate.py              # Dynamic learning rate schedulers
│   ├── weight_decay.py               # Manual weight decay implementation
│   └── dropout_Layer.py 
