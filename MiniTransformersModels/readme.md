# 🤖 MiniTransformers: From-Scratch Implementations

**A deep dive into the internal mechanisms of Transformer architectures.**

This repository contains a collection of **"Mini" Deep Learning architectures** built from the ground up in PyTorch. The goal is to deconstruct modern models (BERT, GPT, DETR, MaskFormer, CLIP, ViT) to understand their core components—Attention mechanisms, Positional Embeddings, Queries, and Patching—without the complexity of massive pre-trained libraries.

The repository covers **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and **Multimodal Learning**.

---

## 📂 Project Structure

| Domain | Model | Class Name | Task | Key Concept |
| :--- | :--- | :--- | :--- | :--- |
| **NLP** | **1. MiniBERT** | `MiniBert` | Sentiment Analysis | Bidirectional Attention & [CLS] Token |
| **NLP** | **2. MiniGPT** | `MiniGPt` | Generative Logic | Causal Masking (Decoder) |
| **CV** | **3. MiniViT** | `ClassificationVit` | Image Classification | Patching & Linear Projection |
| **CV** | **4. MiniDETR** | `DetrModel` | Object Detection | Set Prediction (Queries) |
| **CV** | **5. MiniSeg** | `SegModel` | Segmentation | Mask Embeddings (MaskFormer) |
| **Multi** | **6. MiniCLIP** | `ClipModel` | Image-Text Alignment | Contrastive Loss |

---

## 🧠 Natural Language Processing (NLP)

### 1. MiniBERT: Sentiment Analysis
**File:** `minibert.py`

A proof-of-concept **Encoder-only** architecture. Unlike standard RNNs, this model utilizes bidirectional self-attention to understand the full context of a sentence simultaneously.

* **Mechanism:**
    1.  **Input:** Token Indices.
    2.  **Tokenization:** Adds a special `[CLS]` token to the start of the sequence.
    3.  **Encoder:** Processes the sequence via Self-Attention.
    4.  **Head:** Extracts the vector corresponding to `[CLS]` (index 0) and passes it through an MLP for classification.
* **Result:** Mastered grammar and sentiment rules (100% Accuracy) on synthetic data.

### 2. MiniGPT: Generative Reasoning
**File:** `minigpt.py`

Validates the **Decoder-only** (Autoregressive) architecture. Uses Causal Masking to ensure the model predicts the next token based only on past context.

* **Capabilities:** Arithmetic logic, pattern recognition, and basic Q&A.
* **Architecture:** `Embeddings` $\to$ `GPT Decoder` $\to$ `MLP Head`.

---

## 👁️ Computer Vision (CV)

### 3. MiniViT: Vision Transformer
**File:** `minivit.py` (Class: `ClassificationVit`)

This model adapts the Transformer architecture for vision by treating image patches exactly like words in a sentence.

* **Pipeline:**
    1.  **Patching:** Image sliced into $P \times P$ squares (`image_to_patches`).
    2.  **Projection:** Patches are flattened and projected to vector space.
    3.  **[CLS] Token:** A learnable vector is appended to the patch sequence.
    4.  **Transformer:** Encoder blocks process the sequence.
    5.  **Classification:** Only the `[CLS]` output is used for the final prediction.

### 4. MiniDETR: Object Detection
**File:** `minidetr.py`

An implementation of **DETR (Detection Transformer)**. It removes the need for Anchor Boxes and Non-Maximum Suppression (NMS).

* **Mechanism:**
    * **Backbone:** ResNet50 extracts feature maps.
    * **Transformer:** Processes features with **Learnable Queries**.
    * **Heads:** Two MLPs predict **Bounding Boxes** (Coords) and **Classes** for each query.

### 5. MiniSeg: Semantic Segmentation (MaskFormer)
**File:** `MiniSegmeationMaskFormer.py`

A modern approach to segmentation that treats it as a mask classification problem.

* **Architecture:**
    * **Pixel Decoder:** Generates per-pixel embeddings.
    * **Transformer Decoder:** Updates query embeddings.
    * **Fusion:** Dot product between *Mask Embeddings* and *Pixel Embeddings* creates the final segmentation map.

---

## 📎 Multimodal Learning

### 6. MiniCLIP: Image-Text Alignment
**File:** `miniclip.py`

Tests the ability to map images and text into a shared latent space.

* **Components:**
    * `TextEncoder`: Standard Transformer Encoder.
    * `ImageEncoder`: ViT-style Encoder.
* **Objective:** Maximize Cosine Similarity between matching image-text pairs while minimizing it for non-matching pairs.

---

## 🛠️ Usage & Installation

### Dependencies
```bash
pip install torch torchvision numpy matplotlib pillow
