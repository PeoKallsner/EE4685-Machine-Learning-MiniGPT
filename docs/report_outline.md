# Report Outline — MiniGPT (EE4685 Machine Learning)

Use this document as a template when writing the final project report.
Each section includes a brief description and bullet points of what to cover.

---

## Title

> **MiniGPT: Implementing a Decoder-Only Transformer Language Model from Scratch**

**Authors:** <!-- TODO: add names and student IDs -->

**Course:** EE4685 Machine Learning

**Date:** <!-- TODO: add submission date -->

---

## Abstract

*(~150–200 words)*

Provide a concise summary of:
- What problem is being solved (language modelling with a transformer).
- What was built (MiniGPT: a GPT-style decoder-only transformer in PyTorch).
- Key results (perplexity achieved, number of parameters, training corpus).
- Main conclusions.

---

## 1  Introduction

- Motivation: Why are language models important?
- Brief history of neural language models (RNNs → attention → transformers → GPT).
- Objective: implement and train a small transformer LM from scratch.
- Outline of the report.

<!-- TODO: write this section -->

---

## 2  Background

### 2.1  Language Modelling

- Definition: model the probability distribution over sequences of tokens.
- Next-token prediction: `P(x_t | x_1, ..., x_{t-1})`.

### 2.2  The Transformer Architecture

- Self-attention mechanism.
- Multi-head attention.
- Positional embeddings.
- Feed-forward networks.
- Layer normalisation and residual connections.
- Decoder-only vs. encoder-decoder variants.

### 2.3  Training Objective

- Cross-entropy loss (teacher forcing).
- Perplexity as evaluation metric.

<!-- TODO: write this section, citing relevant papers -->

---

## 3  Model Architecture

Describe the implemented MiniGPT model (refer to `docs/architecture_notes.md`):

- Tokenisation scheme (character-level, vocabulary size).
- Embedding layers (token + positional).
- Transformer blocks: how many, what size.
- LM head.
- Total number of parameters.
- Include the architecture diagram.

<!-- TODO: write this section, include diagram and parameter table -->

---

## 4  Data

- Training corpus: which dataset, how many characters / tokens.
- Pre-processing steps.
- Train / validation / test split sizes.

<!-- TODO: write this section after data collection -->

---

## 5  Training

- Optimiser: AdamW, learning rate, weight decay.
- Batch size, sequence length, number of epochs / steps.
- Learning-rate schedule (if any).
- Hardware used and training time.
- Training and validation loss curves (include figure).

<!-- TODO: write this section after training -->

---

## 6  Results

### 6.1  Quantitative Results

| Model | n_layer | n_embd | n_head | Params | Val. Perplexity |
|-------|---------|--------|--------|--------|----------------|
| MiniGPT (default) | 4 | 128 | 4 | — | — |
| Ablation: 2 layers | 2 | 128 | 4 | — | — |
| Ablation: larger | 6 | 256 | 8 | — | — |

### 6.2  Qualitative Results

Include 3–5 examples of generated text (with different prompts and
temperatures).  Comment on coherence, style, and limitations.

<!-- TODO: write this section after experiments -->

---

## 7  Discussion

- What worked well?
- What didn't work?
- Comparison with expected results from the literature.
- Limitations (small dataset, character-level tokenisation, compute budget).
- Possible future improvements (BPE tokenisation, longer context, weight tying).

<!-- TODO: write this section -->

---

## 8  Conclusion

Summary of findings.  Restate key results and lessons learned.

<!-- TODO: write this section -->

---

## References

<!-- TODO: add all cited papers and resources using a consistent citation style -->

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
   https://arxiv.org/abs/1706.03762

2. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
   OpenAI Blog. https://openai.com/research/better-language-models

3. Karpathy, A. (2022). *nanoGPT*. GitHub.
   https://github.com/karpathy/nanoGPT

---

## Appendix

### A  Hyperparameter Search Details

<!-- TODO: describe any grid/random search experiments -->

### B  Full Training Curves

<!-- TODO: attach additional plots if not included in the main body -->
