# LLM Concepts

<div align="center">

**A practical collection of transformer and LLM building blocks implemented from scratch.**

Explore attention mechanisms, positional encodings, and tokenization with readable code and runnable examples.

[Attention](#attention) · [Positional Encodings](#positional-encodings) · [Tokenization](#tokenization) · [Getting Started](#getting-started)

</div>

---

## What You'll Find

This repository is an educational codebook for understanding the core ideas behind modern large language models. The implementations favor clarity and experimentation over production-scale optimization.

| Area | Concepts |
| --- | --- |
| **Attention** | Self-attention, multi-head attention, causal masking, MQA, GQA, and sparse attention |
| **Positional Encoding** | Sinusoidal, learned, RoPE, and ALiBi-style approaches |
| **Tokenization** | Byte-level BPE, WordPiece, and Uni-Gram language model tokenization |

---

## Repository Structure

```text
LLM-Concepts/
├── Attention/
│   ├── self_attention.py
│   ├── multihead_attention.py
│   ├── Casual_masked_multihead_attention.py
│   ├── Multi_Query_Attention.py
│   ├── group_Query_Attention.py
│   └── sparse_attention.py
├── Positional Encodings/
│   └── simple_positional_encoding.py
└── Tokenization/
    ├── Byte Level BPE/
    ├── Word Piece/
    └── Uni-Gram Language Model Tokenization/
```

---

## Attention

The attention modules show how models learn relationships between tokens and how those relationships are made efficient.

- **Self-Attention** — the fundamental query, key, and value mechanism
- **Multi-Head Attention** — parallel attention projections that capture multiple relationships
- **Causal Masked Attention** — prevents each position from attending to future tokens
- **Multi-Query Attention (MQA)** — shares key and value projections across query heads
- **Grouped-Query Attention (GQA)** — a balanced middle ground between MHA and MQA
- **Sparse Attention** — local, causal, global, strided, random, block, top-k, and hybrid patterns

### Run an example

```powershell
python ".\Attention\multihead_attention.py"
python ".\Attention\Multi_Query_Attention.py"
python ".\Attention\group_Query_Attention.py"
python ".\Attention\sparse_attention.py"
```

---

## Positional Encodings

Because attention itself is order-independent, positional information is essential.

The [positional encoding module](./Positional%20Encodings/simple_positional_encoding.py) includes:

- Sinusoidal positional encoding
- Learned positional embeddings
- Rotary Position Embeddings (RoPE)
- An ALiBi implementation in progress

---

## Tokenization

Tokenization converts raw text into the numeric units consumed by language models.

### Byte-Level BPE

A transparent, educational byte-level Byte Pair Encoding implementation with training, encoding, decoding, special tokens, JSON serialization, and validation.

```powershell
cd ".\Tokenization\Byte Level BPE"
python train.py
python infrence.py
```

The training script reads from `Dataset/input.txt`, trains a tokenizer, saves `tokenizer.json`, and demonstrates encoding and decoding.

### WordPiece

The WordPiece notebook demonstrates tokenizer training with the `tokenizers` library.

### Uni-Gram

The Uni-Gram notebook demonstrates training and loading a SentencePiece unigram model.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shahfahadmle/LLM-Concepts.git
cd LLM-Concepts
```

### 2. Create an environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

The core attention and positional encoding examples use PyTorch:

```powershell
pip install torch
```

Additional notebook dependencies:

```powershell
pip install jupyter tokenizers sentencepiece
```

### 4. Explore

Open a module directly, run a script, or launch Jupyter:

```powershell
jupyter notebook
```

---

## Learning Path

1. Start with **Self-Attention**
2. Move to **Multi-Head and Causal Attention**
3. Compare **MQA and GQA**
4. Experiment with **Sparse Attention**
5. Understand **Positional Encodings**
6. Train and inspect **BPE, WordPiece, and Uni-Gram tokenizers**

---

## Notes

- The repository is intended for learning and experimentation.
- Files contain small runnable demonstrations where applicable.
- Some implementations are still experimental and may continue to evolve.
- This project was built with PyTorch, Jupyter notebooks, `tokenizers`, and SentencePiece.

---

## Contributing

Contributions that improve clarity, correctness, documentation, or examples are welcome.

1. Fork the repository
2. Create a focused feature branch
3. Make your changes
4. Run the relevant examples or notebooks
5. Open a pull request

---

<div align="center">

Built to make modern LLM concepts easier to see, understand, and experiment with.

</div>