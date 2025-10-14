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
├── 📁 core/ # Core engine (manual forward/backward, optimizers, base model)
│ ├── MLP_layer.py # Dense layers & manual linear algebra
│ ├── MLP_initializers.py # Xavier, He Normal, Uniform
│ ├── optimizers.py # SGD, RMSprop, Adam (from scratch)
│ ├── losses.py # CrossEntropy, MSE, BCE
│ ├── metrics.py # Accuracy, Precision, Recall, F1
│ ├── model_structure.py # Base Model class + training loop
│ ├── Dropout_layer.py # Custom dropout
│ └── *.md # Theoretical explanations
│
├── 📁 CNN/
│ ├── 📁 Loop_based_cnn/ # Educational loop-based CNNs
│ │ ├── Cnn_layers.py # Manual convolution, pooling, batchnorm
│ │ ├── Cnn_operations.py # Pixel-by-pixel conv & backprop
│ │ └── Cnn_initializers.py # Kernel initialization
│ │
│ ├── 📁 Vectorised_Cnn_operations/ # Optimized vectorized CNN version
│ │ ├── Vec_cnn_Layers.py
│ │ └── Vectorised_Cnn_operations.py
│ │
│ └── resnet.py # Custom handcrafted ResNet
│
├── 📁 Custom_transformers/
│ ├── transformeroperations.py # Manual multi-head attention, masking, QKV ops
│ └── Encoder_Decoders.py # Encoder/Decoder architecture logic
│
├── 📁 Rnn/
│ ├── RNN_operations.py # Manual RNN/LSTM ops
│ ├── Rnn_Layers.py
│ ├── Rnn_model.py
│ ├── datasets/ # CSV datasets for translation
│ └── notebooks/ # Educational comparison vs PyTorch
│
├── 📁 MiniTransformersModels/
│ ├── MiniGpt.py
│ ├── MiniBert.py
│ ├── MiniVit.py
│ ├── MiniClip.py
│ ├── MiniDetr.py
│ ├── MiniSegmentationMaskFormer.py
│ └── test.ipynb
│
├── 📁 GANs/ # (Planned) Generative Adversarial Networks
│
├── 📁 DeepLearningNotebooks/ # Educational notebooks
│ ├── single_perceptron.ipynb
│ ├── regression_MLP.ipynb
│ ├── Multi_classification_MLP.ipynb
│ ├── loop_based_mnist.ipynb
│ ├── Vec_Cnn_mnist.ipynb
│ └── California_housing.ipynb
│
├── 📁 TransformerModelsNotebooks/ # Transformer training notebooks
│ ├── MiniGpt_notebook.ipynb
│ ├── MiniBert_notebook.ipynb
│ ├── MiniClip_Notebook.ipynb
│ ├── MiniDetr_notebook.ipynb
│ ├── MiniVitClassifier_notebook.ipynb
│ └── MiniSegTransformer.ipynb
│
└── 📁 utils/
├── activations.py # ReLU, Sigmoid, Tanh, Softmax (manual)
├── batch_normalization_Layer.py # Custom BatchNorm
├── data_manipulation.py # Mini data loaders
├── learning_rate.py # LR schedulers
├── weight_decay.py # Manual weight decay  
└── dropout_Layer.py # Custom dropout   

## 🧠 Educational Goals
- Learn **how deep learning models really work**  
- Understand **mathematical operations** behind training  
- Write **manual forward and backward passes**  
- Compare handcrafted implementations with PyTorch’s automatic modules  

---


## 🧑‍💻 Author
📫 [abderraoufheboul@gmail.com]  

⭐ *If you find this project valuable, give it a star and share it — learning deep learning from scratch starts here.*
