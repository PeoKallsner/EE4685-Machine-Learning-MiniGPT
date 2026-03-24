# Project Plan — MiniGPT (EE4685 Machine Learning)

This document outlines the project timeline, milestones, and task breakdown
for the MiniGPT course project.

---

## Objective

Build and train a miniature GPT-style (decoder-only transformer) language
model from scratch using PyTorch.  Evaluate it on a small text corpus and
analyse the results.

---

## Timeline

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | — | Project setup: scaffold, data acquisition, exploratory analysis |
| 2 | — | Implement tokeniser and dataset pipeline |
| 3 | — | Implement attention module and single transformer block |
| 4 | — | Assemble full MiniGPT model, write training loop |
| 5 | — | Train model, monitor loss / perplexity curves |
| 6 | — | Implement generation, qualitative evaluation |
| 7 | — | Hyperparameter experiments (layers, heads, embedding size) |
| 8 | — | Write final report, prepare presentation |

---

## Milestones and Deliverables

### Milestone 1 — Project Scaffold ✅
- [x] Repository structure and placeholder files
- [x] `requirements.txt` and `.gitignore`
- [x] Config files (`configs/`)
- [x] Documentation stubs (`docs/`)
- [x] Test stubs (`tests/`)

### Milestone 2 — Data Pipeline
- [x] Download and inspect raw text corpus
- [ ] Implement `CharTokenizer` (build_vocab, encode, decode, save, load)
- [ ] Implement `TextDataset` (sliding window pairs)
- [ ] Implement `prepare_splits` (train / val / test)
- [ ] Pass all tests in `tests/test_tokenizer.py` and `tests/test_dataset.py`

### Milestone 3 — Attention Mechanism
- [ ] Implement `MultiHeadSelfAttention` with causal mask
- [ ] Verify output shape and causal property (tests)
- [ ] Understand the maths: QKV projections, scaled dot-product

### Milestone 4 — Full Model
- [ ] Implement `FeedForward` (MLP) block
- [ ] Implement `TransformerBlock` (attention + FFN + residuals + LayerNorm)
- [ ] Assemble `MiniGPT` (embeddings + N blocks + LM head)
- [ ] Pass all tests in `tests/test_model.py`

### Milestone 5 — Training
- [ ] Implement training loop (`src/train.py`)
- [ ] Implement checkpoint save / load
- [ ] Train on chosen corpus, plot loss / perplexity curves
- [ ] Tune hyperparameters (at least one ablation)

### Milestone 6 — Generation & Evaluation
- [ ] Implement autoregressive generation (`src/generate.py`)
- [ ] Implement perplexity evaluation (`src/evaluate.py`)
- [ ] Generate sample text, review qualitatively

### Milestone 7 — Report
- [ ] Fill in report sections (see `docs/report_outline.md`)
- [ ] Include training curves, perplexity results, sample outputs
- [ ] Discussion of design decisions and limitations

---

## Team

<!-- TODO: add team member names and student IDs -->

| Name | Student ID | Responsibilities |
|------|-----------|-----------------|
| — | — | — |

---

## Notes

<!-- TODO: add any project-specific notes, constraints, or decisions here -->
