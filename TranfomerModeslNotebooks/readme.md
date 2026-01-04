# 🤖 MiniTransformers: From-Scratch Implementations

**A deep dive into the internal mechanisms of Transformer architectures.**

This repository contains a collection of **"Mini" Deep Learning architectures** built from the ground up in PyTorch. The goal is to deconstruct modern models (BERT, GPT, ViT, MaskFormer) to understand their core components—Attention mechanisms, Positional Embeddings, and Patching—without the complexity of massive pre-trained libraries.

The repository covers **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and **Multimodal Learning**.

---

## 📂 Project Structure

| Model | Type | Task | Key Concept |
| :--- | :--- | :--- | :--- |
| **1. MiniBERT** | Encoder | Sentiment Analysis | Bidirectional Attention |
| **2. MiniGPT** | Decoder | Generative Logic | Causal Masking |
| **3. MiniCLIP** | Multimodal | Image-Text Alignment | Contrastive Loss |
| **4. MiniSeg** | Vision | Segmentation | Dense Prediction |
| **5. MiniViT** | Vision | Classification | Patch Embeddings |

---

## 🎭 1. MiniBERT: Sentiment Analysis
**File:** `MiniBert_notebook.ipynb`

A proof-of-concept implementation of the **Encoder-only** architecture. Unlike standard RNNs, this model utilizes bidirectional self-attention to understand the full context of a sentence simultaneously.

### **The Experiment**
* **Data:** Synthetic Sentiment Generator (61-token vocabulary).
* **Input:** Sentences like *"The service was really amazing"* vs *"I regret this book"*.
* **Task:** Binary Classification (Positive/Negative).

### **Results**
The model mastered the grammar and sentiment rules perfectly, validating that the Self-Attention mechanism correctly attends to keywords (e.g., "amazing", "hate") regardless of their position.

| Epoch | Train Loss | Validation Accuracy | Status |
| :--- | :--- | :--- | :--- |
| 1 | 0.9774 | 100% | Converged |
| **10** | **0.4841** | **100%** | **Perfect** |

---

## 🦜 2. MiniGPT: Generative Reasoning
**File:** `MiniGpt_notebook.ipynb`

This notebook validates the **Decoder-only** (Autoregressive) architecture. To prove the engine works, the model was subjected to 5 distinct "Cognitive Unit Tests" to verify it learns logic, not just probability.

### **The 5 Unit Tests**
1.  **Arithmetic:** Completing number sequences (`2, 4, 6` $\to$ `8`).
2.  **Grammar:** Generating Subject-Verb-Adjective structures.
3.  **Pattern Recognition:** Learning repeating symbols (`A, B, A, B...`).
4.  **Memory (Reversal):** Reading a sequence and generating it backward.
5.  **Context Retrieval (QA):** Answering questions based on learned rules (`Color of sky?` $\to$ `Blue`).

### **Results**
The Causal Masking worked correctly, preventing the model from "peeking" at future tokens.

| Task | Initial Loss | Final Loss | Observation |
| :--- | :--- | :--- | :--- |
| **Arithmetic** | 0.1031 | **0.0245** | Learned numerical progression. |
| **QA (Chatbot)** | 0.1425 | **0.0259** | Learned entity-attribute mapping. |
| **Reversal** | 0.1455 | 0.1078 | Hardest task (requires long-term positional memory). |

---

## 📎 3. MiniCLIP: Multimodal Alignment
**File:** `MiniClip_Notebook.ipynb`

An experimental implementation of **CLIP (Contrastive Language-Image Pre-Training)**. This tests the framework's ability to process two distinct modalities simultaneously and align their latent representations.

* **Data:** A "Mini-World" of synthetic geometric shapes paired with text descriptions ("red ball", "blue square").
* **Mechanism:** Two encoders (Image & Text) optimized via Contrastive Loss to maximize the similarity of correct pairs.
* **Status:** *Experimental*. Demonstrates the difficulty of training multimodal models from scratch without large batch sizes.

---

## 🖼️ 4. MiniSeg: Semantic Segmentation
**File:** `MiniSegTransformer.ipynb`

This tests the framework on **Dense Prediction** using **Real Data** (OxfordIIITPet Dataset). Unlike classification (one label per image), this model must classify **every pixel**.

### **Architecture: MaskFormer-Lite**
* **Input:** 32x32 RGB Images.
* **Output:** 8x8 Segmentation Maps (4 Classes).
* **Mechanism:** Uses a Transformer to process image patches and project them into class masks.

### **Results**
The model converged extremely fast on the Oxford Pets dataset, proving that Transformers can handle high-dimensional spatial outputs.

| Epoch | Train Loss | Validation Accuracy | Note |
| :--- | :--- | :--- | :--- |
| 1 | 4.55 | 78.88% | Rapid initial learning. |
| **10** | **1.04** | **80.44%** | **Stable convergence.** |

---

## 👁️ 5. MiniViT: Vision Transformer Classifier
**File:** `MiniVitClassifier_notebook.ipynb`

This module implements the **Pure Vision Transformer (ViT)**. It treats image patches exactly like words in a sentence, entirely removing Convolutions (CNNs).

### **The Mechanism**
1.  **Patching:** Image sliced into $4 \times 4$ squares.
2.  **Projection:** Flattened and projected to vector space.
3.  **Transformer:** Standard Encoder blocks process the sequence.

### **Results & Analysis: The "Inductive Bias" Problem**
I tested this on synthetic datasets (Digits, Shapes). Unlike CNNs, which converge instantly on these tasks, the MiniViT struggled. This result is scientifically significant: it demonstrates the **"Data Hunger"** of Transformers. Lacking the "Inductive Bias" of CNNs (knowing that adjacent pixels are related), ViTs require massive data to learn spatial structure.

| Task | Epoch 1 Acc | Epoch 10 Acc | Interpretation |
| :--- | :--- | :--- | :--- |
| **Digits** | 6.00% | **21.00%** | Slowly learning structure (Random is 10%). |
| **Shapes** | 19.00% | 12.00% | Struggling to distinguish geometry without edges. |

---

