# 🤖 MiniTransformers: From-Scratch Implementations

**A deep dive into the internal mechanisms of Transformer
architectures.**

This repository contains a collection of **"Mini" Deep Learning
architectures** built from the ground up in PyTorch. The goal is to
deconstruct modern models (BERT, GPT, DETR, MaskFormer, CLIP, ViT) to
understand their core components---Attention mechanisms, Positional
Embeddings, Queries, and Patching---without the complexity of massive
pre-trained libraries.

The repository covers **Natural Language Processing (NLP)**, **Computer
Vision (CV)**, and **Multimodal Learning**.

------------------------------------------------------------------------

## 📂 Project Structure

  ------------------------------------------------------------------------------------
  Domain         Model          Class Name            Task             Key Concept
  -------------- -------------- --------------------- ---------------- ---------------
  **NLP**        **1.           `MiniBert`            Sentiment        Bidirectional
                 MiniBERT**                           Analysis         Attention &
                                                                       \[CLS\] Token

  **NLP**        **2. MiniGPT** `MiniGPt`             Generative Logic Causal Masking
                                                                       (Decoder)

  **CV**         **3. MiniViT** `ClassificationVit`   Image            Patching &
                                                      Classification   Linear
                                                                       Projection

  **CV**         **4.           `DetrModel`           Object Detection Set Prediction
                 MiniDETR**                                            (Queries)

  **CV**         **5. MiniSeg** `SegModel`            Segmentation     Mask Embeddings
                                                                       (MaskFormer)

  **Multi**      **6.           `ClipModel`           Image-Text       Contrastive
                 MiniCLIP**                           Alignment        Loss
  ------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Natural Language Processing (NLP)

### 1. MiniBERT: Sentiment Analysis

**File:** `minibert.py`

A proof-of-concept **Encoder-only** architecture using bidirectional
self-attention.

**Key idea:** a special `[CLS]` token is prepended, and its final
embedding is used for classification.

------------------------------------------------------------------------

### 2. MiniGPT: Generative Reasoning

**File:** `minigpt.py`

A **Decoder-only** Transformer with causal masking to prevent
information leakage from future tokens.

------------------------------------------------------------------------

## 👁️ Computer Vision (CV)

### 3. MiniViT: Vision Transformer

**File:** `minivit.py` (Class: `ClassificationVit`)

Images are split into patches and treated like tokens.

Pipeline: 1. Patch extraction 2. Linear projection 3. `[CLS]` token
concatenation 4. Transformer encoder 5. Classification from `[CLS]`

------------------------------------------------------------------------

### 4. MiniDETR: Object Detection

**File:** `minidetr.py`

DETR-style architecture using learnable object queries instead of anchor
boxes.

------------------------------------------------------------------------

### 5. MiniSeg: Semantic Segmentation

**File:** `MiniSegmeationMaskFormer.py`

MaskFormer-inspired segmentation using mask embeddings and pixel
embeddings.

------------------------------------------------------------------------

## 📎 Multimodal Learning

### 6. MiniCLIP: Image-Text Alignment

**File:** `miniclip.py`

Maps images and text into a shared embedding space using contrastive
learning.

------------------------------------------------------------------------

## 🛠️ Usage & Installation

### Dependencies

``` bash
pip install torch torchvision numpy matplotlib pillow
```

------------------------------------------------------------------------

### Quick Start

#### Vision Transformer

``` python
from minivit import ClassificationVit

model = ClassificationVit(
    optimizer="adam",
    loss="Crossentropy",
    classes=10,
    dmodel=128,
    patch_size=4,
    vocab_size=100
)
```

#### MiniBERT

``` python
from minibert import MiniBert

model = MiniBert(
    optimizer="adam",
    loss="Crossentropy",
    vocab_size=1000,
    dmodel=128,
    nclasses=2
)
```

#### MiniDETR

``` python
from minidetr import DetrModel

model = DetrModel(
    optimizer="adam",
    loss="Crossentropy",
    dmodel=64,
    Nqueries=10,
    nclasses=4,
    numchanels=2048
)

pred_boxes, pred_classes = model.forward_propagation(image_tensor)
```

------------------------------------------------------------------------

## ✅ Philosophy

This repository is **educational by design**: readability, explicit
math, and transparency are prioritized over speed and scalability.
