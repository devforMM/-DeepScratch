🤖 NeuroCore Transformers: Attention Is All You Code
NeuroCore Transformers is a complete, from-scratch implementation of the Transformer architecture. It deconstructs the "Black Box" of modern NLP and Vision models by implementing the mathematically rigorous Scaled Dot-Product Attention mechanism manually.

📂 Module Architecture
The architecture is modular, moving from atomic mathematical operations to full model orchestration:

<img width="883" height="185" alt="image" src="https://github.com/user-attachments/assets/c78ea2d8-203d-4151-a0d6-43eadd82772e" />

🧠 1. The Core Logic: Heads.py
This file contains the "brain" of the Transformer. Instead of using torch.nn.MultiheadAttention, I have implemented the attention formula manually:

<img width="555" height="132" alt="image" src="https://github.com/user-attachments/assets/cc1389cd-04c6-4c45-b1c3-7cb8d11d54af" />

Features:

Manual QKV Projection: Explicit creation of Query, Key, and Value matrices.

Scaled Dot-Product: Manual matrix multiplication and scaling.

Causal Masking: Implementation of the lower-triangular mask (torch.tril) for GPT-style generation (Decoder).

Code vs Math:

<img width="881" height="417" alt="image" src="https://github.com/user-attachments/assets/c017327a-4402-47e4-b5b2-219e21f8cf7d" />

🧱 2. The Architects: Encoder_Decoders.py

This module assembles the atomic heads into functional blocks. It demonstrates a unified approach to Deep Learning where the same blocks build different models.

Encoder: The standard BERT-style block. Stacks Multi_head_attention + Feed_Forward + AddNorm.

Decoder: The standard Translation block. Includes Cross-Attention to attend to the Encoder's output.

DecoderOnly: The GPT block. Uses only Masked Self-Attention (no Cross-Attention).

VitEncoder: A specialized encoder for Vision Transformers, proving the architecture handles image patches just like words.

DetrDecoder: A specialized decoder for Object Detection (DETR), handling object queries.

🎛️ 3. The Orchestrator: AttentionModel.py

A flexible wrapper that creates any Transformer variant on the fly. It manages the training loop, loss computation, and forward propagation.

Supported Model Types:
EncoderDecoder: For Sequence-to-Sequence tasks (Translation).

EncoderOnly: For Classification/Understanding (BERT, ViT).

DecoderOnly: For Generative AI (GPT).

🚀 Key Technical Achievements

Cross-Attention Implementation: Successfully implemented logic where Queries come from the Decoder, but Keys/Values come from the Encoder (cross_Multi_head_attention_layer).

Unified Framework: The same code powers MiniGPT (Text), MiniViT (Images), and MiniDETR (Object Detection).

Manual Weight Management: The self.weights list manually tracks parameters across the entire deep structure for the custom optimizer to update.

📜 License
MIT License. Built strictly for educational and research purposes.
