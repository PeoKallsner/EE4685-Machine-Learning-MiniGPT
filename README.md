# EE4685 — Machine Learning: MiniGPT

A course project implementing a **decoder-only transformer language model** (MiniGPT) from scratch using PyTorch, as part of the EE4685 Machine Learning course.

---

## Project Overview

MiniGPT is a miniature GPT-style language model trained on a small text corpus. The goal is to understand the core components of modern large language models by building one piece-by-piece:

- Tokenisation (character-level or BPE)
- Embedding layers (token + positional)
- Multi-head self-attention with causal masking
- Transformer decoder blocks
- Language-model head for next-token prediction
- Autoregressive text generation

---

## Repository Structure

```
EE4685-Machine-Learning-MiniGPT/
├── configs/
│   ├── default_config.yaml   # Training hyperparameters
│   └── model_config.yaml     # Model architecture settings
├── data/
│   ├── raw/                  # Raw text data (not tracked by git)
│   ├── processed/            # Tokenised / pre-processed data
│   └── README.md             # Data acquisition instructions
├── docs/
│   ├── architecture_notes.md # Transformer architecture notes
│   ├── project_plan.md       # Project timeline and milestones
│   └── report_outline.md     # Report structure outline
├── notebooks/
│   └── exploration.ipynb     # Exploratory data analysis notebook
├── src/
│   ├── __init__.py
│   ├── attention.py          # Multi-head self-attention module
│   ├── config.py             # Configuration dataclass
│   ├── dataset.py            # PyTorch Dataset for language modelling
│   ├── evaluate.py           # Evaluation utilities (perplexity, etc.)
│   ├── generate.py           # Autoregressive text generation
│   ├── main.py               # Entry point: train or generate
│   ├── model.py              # Full MiniGPT model
│   ├── tokenizer.py          # Tokeniser (character-level)
│   ├── train.py              # Training loop
│   └── utils.py              # Miscellaneous helper functions
├── tests/
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_tokenizer.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PeoKallsner/EE4685-Machine-Learning-MiniGPT.git
cd EE4685-Machine-Learning-MiniGPT
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare data

See [data/README.md](data/README.md) for instructions on downloading and preprocessing a text corpus.

### 5. Train the model

```bash
python src/main.py --mode train --config configs/default_config.yaml
```

### 6. Generate text

```bash
python src/main.py --mode generate --prompt "Once upon a time"
```

---

## Running Tests

```bash
pytest tests/
```

---

## Configuration

Training and model hyperparameters are controlled via YAML config files in the `configs/` directory:

| File | Purpose |
|------|---------|
| `configs/default_config.yaml` | Training settings (learning rate, batch size, epochs, …) |
| `configs/model_config.yaml` | Architecture settings (layers, heads, embedding size, …) |

---

## Documentation

Additional documentation is available in the `docs/` directory:

- [Project Plan](docs/project_plan.md) — timeline and milestones
- [Architecture Notes](docs/architecture_notes.md) — transformer design decisions
- [Report Outline](docs/report_outline.md) — structure for the final report

---

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Key libraries:

- **PyTorch** — deep learning framework
- **PyYAML** — config file parsing
- **NumPy** — numerical utilities
- **tqdm** — progress bars
- **pytest** — testing

---

## License

This project is created for academic purposes as part of the EE4685 Machine Learning course.
