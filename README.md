<div align="center">

<br/>

```
██╗     ██╗     ███╗   ███╗    ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗
██║     ██║     ████╗ ████║    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝
██║     ██║     ██╔████╔██║    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗
██║     ██║     ██║╚██╔╝██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝
███████╗███████╗██║ ╚═╝ ██║    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗
╚══════╝╚══════╝╚═╝     ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝
```

# LLM Data Preprocessing, Embedding & Architecture

**A comprehensive hands-on Jupyter notebook exploring the foundational internals of Large Language Models — from raw text to token embeddings to Transformer architecture.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co/transformers/)
[![tiktoken](https://img.shields.io/badge/tiktoken-OpenAI-412991?style=flat-square)](https://github.com/openai/tiktoken)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Notebook](https://img.shields.io/badge/Open-Notebook-orange?style=flat-square&logo=jupyter)](./LLM_Data_Preprocessing.ipynb)

<br/>

> *"To understand how large language models think, you must first understand how they see."*

<br/>

</div>

---

## 📖 Overview

This repository contains a structured, educational deep-dive into the **data pipeline and internal mechanics that power modern Large Language Models (LLMs)** such as GPT, LLaMA, and BERT.

Rather than treating LLMs as black boxes, this notebook tears them open — walking through every stage from raw corpus to the final embedding space. It is built as both a **learning resource** and a **reference implementation** for anyone who wants to understand what truly happens before a model ever sees a forward pass.

The notebook is organized around three core pillars:

| Pillar | What You'll Build | Concepts Covered |
|--------|-------------------|------------------|
| 🧹 **Data Preprocessing** | A clean, structured NLP pipeline | Tokenization, normalization, BPE, sliding windows |
| 🔢 **Embeddings** | Token + positional embedding layers | Word2Vec, token IDs, positional encodings |
| 🏗️ **Architecture** | GPT-style Transformer components | Self-attention, multi-head attention, feed-forward layers |

---

## 🎯 What This Notebook Covers

### Stage 1 — Raw Text Preprocessing

The journey begins with unstructured text and ends with a clean, model-ready corpus.

```
Raw Text  →  Normalization  →  Tokenization  →  Token IDs  →  Sliding Window Batches
```

- **Text normalization** — lowercasing, punctuation handling, Unicode cleanup
- **Tokenization strategies** — character-level, word-level, and subword comparison
- **Byte Pair Encoding (BPE)** — the algorithm behind GPT's `tiktoken` tokenizer
- **Vocabulary construction** — building and mapping a `{token: id}` lookup table
- **Sliding window dataset** — generating `(input, target)` sequence pairs with configurable `context_length` and `stride`

---

### Stage 2 — Embedding Layer

Converts discrete token IDs into continuous, trainable vector representations.

```
Token IDs  →  Token Embedding Matrix  →  + Positional Encoding  →  Embedded Sequence
```

- **Token embeddings** — a learnable `nn.Embedding(vocab_size, d_model)` layer mapping each token to a dense vector
- **Positional encodings** — both learned (GPT-style) and sinusoidal approaches, injecting sequence-order information
- **Embedding visualization** — how semantically similar tokens cluster in the embedding space
- **Combined input representation** — the final `token_embedding + position_embedding` tensor fed into the Transformer

---

### Stage 3 — Transformer Architecture

Implements the core building blocks of a decoder-only (GPT-style) Transformer from scratch in PyTorch.

```
Embedded Input  →  Multi-Head Self-Attention  →  FFN  →  Layer Norm  →  Residual  →  Output Logits
```

- **Scaled dot-product attention** — `softmax(QKᵀ / √d_k) · V` implemented step-by-step
- **Causal masking** — ensuring each token only attends to previous positions during text generation
- **Multi-head attention** — splitting queries, keys, and values across `n_head` independent attention heads
- **Feed-forward sublayer** — the expand-then-compress MLP with GELU activation
- **Layer normalization & residual connections** — stabilizing gradients in deep networks
- **Stacked Transformer blocks** — assembling the full model from individual composable components

---

## 🗂️ Repository Structure

```
LLM-Data-Preprocessing-Embedding-Architecture-/
│
├── 📓 LLM_Data_Preprocessing.ipynb   ← Main notebook (all 3 stages)
├── 📄 README.md                       ← This file
├── 🚫 .gitignore
└── ⚙️  .gitattributes
```

The entire implementation lives in a single well-structured notebook with clearly separated sections, markdown explanations for every concept, and inline code comments throughout.

---

## 🚀 Quick Start

### Prerequisites

Make sure you have **Python 3.9+** and **pip** installed.

### 1 — Clone the repository

```bash
git clone https://github.com/Minhaj078/LLM-Data-Preprocessing-Embedding-Architecture-.git
cd LLM-Data-Preprocessing-Embedding-Architecture-
```

### 2 — Install dependencies

```bash
# Core ML stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# NLP + notebook tools
pip install transformers tiktoken numpy matplotlib scikit-learn tqdm jupyterlab
```

> **CPU-only machines:** Replace the PyTorch install URL with `https://download.pytorch.org/whl/cpu`

### 3 — Launch the notebook

```bash
jupyter lab LLM_Data_Preprocessing.ipynb
```

Or with classic Jupyter:

```bash
jupyter notebook LLM_Data_Preprocessing.ipynb
```

### 4 — Run all cells

Use `Kernel → Restart & Run All` to execute the full pipeline end-to-end.

---

## 🧠 Key Concepts Explained

### Byte Pair Encoding (BPE)

BPE is the subword tokenization algorithm used by GPT-2, GPT-3, GPT-4, and most modern LLMs. It starts with character-level tokens and iteratively merges the most frequent adjacent pairs until a target vocabulary size is reached.

```
"tokenization" → ["token", "ization"]    # BPE splits at learned boundaries
"untokenized"  → ["un", "token", "ized"] # Handles unseen words gracefully
```

This approach allows the model to handle rare words, out-of-vocabulary terms, and new languages without collapsing to generic `[UNK]` tokens.

---

### Why Positional Encodings?

The self-attention mechanism is inherently **position-agnostic** — it sees a set of tokens, not an ordered sequence. Positional encodings inject order information directly into each token's embedding vector.

```python
# Sinusoidal (fixed, from the original "Attention Is All You Need" paper)
PE(pos, 2i)   = sin(pos / 10000 ^ (2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000 ^ (2i / d_model))

# Learned (GPT-style, trainable parameters)
position_embedding = nn.Embedding(context_length, d_model)
```

---

### Causal Masking in Decoder-Only Models

For text generation, a model must only use past tokens to predict the next one — it cannot "cheat" by looking ahead. The causal mask is an upper-triangular matrix of `-inf` values applied before the softmax:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
attention_scores = attention_scores + mask
# Positions above the diagonal become -inf → softmax → 0 attention weight
```

---

### Scaled Dot-Product Attention

The core operation of every Transformer block:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

The `√d_k` scaling term prevents dot products from growing too large in high-dimensional spaces, which would push the softmax into regions with near-zero gradients — making training unstable.

---

### Multi-Head Attention

Rather than computing one set of attention weights, multi-head attention runs `h` independent attention functions in parallel, each with its own learned projection matrices:

```python
# Each head learns to attend to different types of relationships
head_1: attends to syntactic dependencies  (subject-verb agreement)
head_2: attends to semantic relationships  (word meaning similarity)
head_3: attends to positional patterns     (phrase boundaries)
...
# Outputs are concatenated and projected back to d_model dimensions
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **PyTorch 2.x** | Tensor operations, `nn.Module` implementations |
| **tiktoken** | OpenAI's fast BPE tokenizer |
| **HuggingFace Transformers** | Reference models and tokenizers |
| **NumPy** | Numerical operations and array manipulation |
| **Matplotlib** | Embedding space visualizations |
| **Jupyter Lab** | Interactive notebook environment |

---

## 📊 Learning Path

This notebook fits naturally into the following progression:

```
[ Python & NumPy Basics ]
          ↓
[ NLP Fundamentals — Bag of Words, TF-IDF ]
          ↓
[ THIS NOTEBOOK — Tokenization · Embeddings · Transformer Internals ]
          ↓
[ Fine-tuning Pre-trained LLMs with HuggingFace ]
          ↓
[ Building RAG Systems with Vector Databases ]
          ↓
[ LoRA / QLoRA for Efficient Fine-tuning ]
```

**Recommended prerequisites:**
- Basic Python and NumPy
- High school linear algebra (matrix multiplication, dot products)
- Familiarity with what a neural network is (loss function, backpropagation)

**What to explore next:**
- Fine-tuning pre-trained models with HuggingFace `Trainer` API
- Building a RAG pipeline with Pinecone or Chroma as the vector store
- Training a small GPT-2-scale model from scratch using the pipeline built here

---

## 📚 References & Further Reading

| Resource | Description |
|----------|-------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The original Transformer paper — Vaswani et al., 2017 |
| [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Language models as unsupervised multitask learners |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Jay Alammar's definitive visual walkthrough |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | Harvard NLP — code-first Transformer implementation |
| [tiktoken](https://github.com/openai/tiktoken) | OpenAI's fast BPE tokenizer |
| [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | Sebastian Raschka's companion repository |
| [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) | End-to-end NLP course with Transformers |

---

## 👨‍💻 Author

<div align="center">

**Minhaj**

[![GitHub](https://img.shields.io/badge/GitHub-Minhaj078-181717?style=flat-square&logo=github)](https://github.com/Minhaj078)

*Python & Full Stack Development · Lovely Professional University*

</div>

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, adapt, and build upon this work with attribution.

---

<div align="center">

**If this notebook helped you understand LLMs better, give it a ⭐ — it helps others find it too.**

<br/>

*Built with curiosity and an unreasonable number of print statements.*

</div>
