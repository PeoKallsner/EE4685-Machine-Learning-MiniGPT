# Data Directory

This directory holds the text data used to train and evaluate **MiniGPT**.

---

## Directory Structure

```
data/
├── raw/          ← Original, unmodified text files go here
├── processed/    ← Tokenised / pre-processed data goes here
└── README.md     ← This file
```

---

## Acquiring Raw Data

For a course project, any plain-text English corpus works well. Some freely
available options are listed below.

### Option A — Tiny Shakespeare (recommended for quick experiments)

```bash
wget -P data/raw/ https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Option B — Project Gutenberg

Download one or more books from <https://www.gutenberg.org/> in plain-text
format and save them to `data/raw/`.

### Option C — Custom corpus

Place any `.txt` file containing your chosen training text in `data/raw/`.

---

## Pre-processing

After placing raw data in `data/raw/`, run the dataset preparation script:

```bash
python src/dataset.py --prepare
```

This will:
1. Read the raw text file(s).
2. Build a vocabulary using the tokeniser (`src/tokenizer.py`).
3. Split the data into train / validation / test sets.
4. Save the processed splits to `data/processed/`.

---

## Notes

- `data/raw/` and `data/processed/` are excluded from version control via
  `.gitignore` to avoid committing large binary or text files.
- Keep the `.gitkeep` placeholder files so the directory structure is
  preserved in git.
