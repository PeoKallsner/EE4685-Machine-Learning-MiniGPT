#!/usr/bin/env python3
"""
Bayesian Character-Level N-gram Language Model

Implements a simple, interpretable baseline for comparison with neural models like MiniGPT.

Key Components:
- Character-level n-gram modeling
- Dirichlet prior smoothing for robust probability estimation
- Perplexity evaluation on train/val/test splits

This model is interpretable and works well for character-level tasks where
neural models may overfit or require more data.
"""

import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import math


# ============================================================================
# 1. VOCABULARY BUILDING
# ============================================================================

class CharVocabulary:
    """Build and manage character-level vocabulary."""
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def build_from_text(self, text: str) -> None:
        """Build vocabulary from unique characters in text.
        
        Args:
            text: Raw text string
        """
        unique_chars = sorted(set(text))
        self.char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(unique_chars)
        print(f"Built vocabulary with {self.vocab_size} unique characters")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs.
        
        Args:
            text: String to encode
            
        Returns:
            List of integer token IDs
        """
        return [self.char_to_id[char] for char in text if char in self.char_to_id]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text.
        
        Args:
            token_ids: List of integer IDs
            
        Returns:
            Decoded string
        """
        return ''.join(self.id_to_char[idx] for idx in token_ids if idx in self.id_to_char)


# ============================================================================
# 2. N-GRAM COUNTING
# ============================================================================

class NGramCounter:
    """Count n-gram occurrences in token sequence."""
    
    def __init__(self, n: int):
        """Initialize counter for n-grams.
        
        Args:
            n: Order of n-gram (e.g., 3 for trigram)
        """
        self.n = n
        self.context_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
    
    def count_ngrams(self, token_ids: List[int]) -> None:
        """Count all n-grams in sequence.
        
        For trigram (n=3):
        - Context = previous 2 characters
        - Prediction = next character
        
        Args:
            token_ids: Sequence of token IDs
        """
        context_len = self.n - 1
        
        for i in range(context_len, len(token_ids)):
            # Extract context (previous n-1 tokens)
            context = tuple(token_ids[i - context_len:i])
            # Current token is what we're predicting
            target = token_ids[i]
            
            # Update counts
            self.context_counts[context][target] += 1
            self.context_totals[context] += 1
        
        print(f"Counted {sum(self.context_totals.values()):,} n-grams")
        print(f"Found {len(self.context_counts)} unique contexts")
    
    def get_counts(self, context: Tuple[int, ...]) -> Counter:
        """Get character counts for a context.
        
        Args:
            context: Tuple of preceding token IDs
            
        Returns:
            Counter of {token_id: count} for this context
        """
        return self.context_counts[context]
    
    def get_context_total(self, context: Tuple[int, ...]) -> int:
        """Get total count for a context.
        
        Args:
            context: Tuple of preceding token IDs
            
        Returns:
            Total number of observations for this context
        """
        return self.context_totals[context]


# ============================================================================
# 3. BAYESIAN N-GRAM LANGUAGE MODEL
# ============================================================================

class BayesianNGramModel:
    """Bayesian character-level n-gram language model with Dirichlet smoothing."""
    
    def __init__(self, vocab_size: int, n: int = 3, alpha: float = 1.0):
        """Initialize Bayesian n-gram model.
        
        Args:
            vocab_size: Size of vocabulary
            n: Order of n-gram (default: 3 for trigram)
            alpha: Dirichlet prior parameter (higher = more smoothing)
        """
        self.vocab_size = vocab_size
        self.n = n
        self.alpha = alpha
        self.counter = NGramCounter(n)
        
        print(f"\nInitialized Bayesian {n}-gram model")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Dirichlet alpha: {alpha}")
    
    def fit(self, token_ids: List[int]) -> None:
        """Fit model by counting n-grams.
        
        Args:
            token_ids: Sequence of token IDs for training
        """
        print(f"\nFitting model on {len(token_ids):,} tokens...")
        self.counter.count_ngrams(token_ids)
    
    def predict_prob(self, context: Tuple[int, ...], target: int) -> float:
        """Compute P(target | context) using Bayesian smoothing.
        
        Formula:
            P(c | h) = (Count(h, c) + alpha) / (Count(h) + alpha * V)
        
        Where:
            - Count(h, c) = number of times c follows context h
            - Count(h) = total observations of context h
            - alpha = Dirichlet prior parameter
            - V = vocabulary size
        
        Args:
            context: Tuple of preceding token IDs
            target: Target token ID to predict probability for
            
        Returns:
            Probability P(target | context)
        """
        count_hc = self.counter.get_counts(context).get(target, 0)
        count_h = self.counter.get_context_total(context)
        
        # Bayesian smoothing
        numerator = count_hc + self.alpha
        denominator = count_h + self.alpha * self.vocab_size
        
        return numerator / denominator
    
    def compute_log_likelihood(self, token_ids: List[int]) -> float:
        """Compute log-likelihood of sequence.
        
        Args:
            token_ids: Sequence of tokens
            
        Returns:
            Sum of log probabilities: sum(log P(c_t | context_t))
        """
        context_len = self.n - 1
        log_prob_sum = 0.0
        
        for i in range(context_len, len(token_ids)):
            context = tuple(token_ids[i - context_len:i])
            target = token_ids[i]
            
            prob = self.predict_prob(context, target)
            # Avoid log(0)
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                log_prob_sum += math.log(1e-10)  # Small constant
        
        return log_prob_sum
    
    def compute_perplexity(self, token_ids: List[int]) -> float:
        """Compute perplexity of sequence.
        
        Perplexity = exp(-1/N * sum(log P(c_t | context_t)))
        
        Lower perplexity = better model
        Perplexity ≈ vocabulary_size = random baseline
        
        Args:
            token_ids: Sequence of tokens
            
        Returns:
            Perplexity score
        """
        context_len = self.n - 1
        if len(token_ids) <= context_len:
            return float('inf')
        
        log_likelihood = self.compute_log_likelihood(token_ids)
        # Number of predictions (tokens after context_len)
        num_predictions = len(token_ids) - context_len
        
        # PPL = exp(-avg_log_prob)
        avg_log_prob = log_likelihood / num_predictions
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def sample_next_token(self, context: Tuple[int, ...]) -> int:
        """Sample next token from predictive distribution.
        
        Args:
            context: Preceding tokens
            
        Returns:
            Sampled token ID
        """
        # Compute probabilities for all tokens
        probs = np.array([
            self.predict_prob(context, token_id)
            for token_id in range(self.vocab_size)
        ])
        
        # Normalize
        probs = probs / probs.sum()
        
        # Sample
        return np.random.choice(self.vocab_size, p=probs)


# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

def split_data(text: str, vocab: CharVocabulary, train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """Split text into train/val/test sets.
    
    Args:
        text: Raw text to split
        vocab: Vocabulary for encoding
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    # Encode text
    token_ids = vocab.encode(text)
    
    # Calculate split points
    n = len(token_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_ids = token_ids[:train_end]
    val_ids = token_ids[train_end:val_end]
    test_ids = token_ids[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_ids):,} tokens (80%)")
    print(f"  Val:   {len(val_ids):,} tokens (10%)")
    print(f"  Test:  {len(test_ids):,} tokens (10%)")
    
    return train_ids, val_ids, test_ids


def compute_random_baseline(vocab_size: int) -> float:
    """Compute perplexity of random character baseline.
    
    For uniform random distribution over vocabulary,
    PPL = vocabulary_size
    
    Args:
        vocab_size: Size of vocabulary
        
    Returns:
        Perplexity of random baseline
    """
    return float(vocab_size)


# ============================================================================
# 5. MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation pipeline."""
    
    print("=" * 70)
    print("BAYESIAN CHARACTER-LEVEL N-GRAM LANGUAGE MODEL")
    print("=" * 70)
    
    # Load Shakespeare data
    data_path = Path("data/raw/input.txt")
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found")
        return
    
    raw_text = data_path.read_text()
    print(f"\nLoaded dataset: {len(raw_text):,} characters")
    
    # Build vocabulary
    print("\n" + "=" * 70)
    print("STEP 1: BUILD VOCABULARY")
    print("=" * 70)
    
    vocab = CharVocabulary()
    vocab.build_from_text(raw_text)
    
    # Split data
    print("\n" + "=" * 70)
    print("STEP 2: SPLIT DATA")
    print("=" * 70)
    
    train_ids, val_ids, test_ids = split_data(raw_text, vocab)
    
    # Train trigram model
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN TRIGRAM MODEL (n=3, alpha=1.0)")
    print("=" * 70)
    
    trigram_model = BayesianNGramModel(vocab.vocab_size, n=3, alpha=1.0)
    trigram_model.fit(train_ids)
    
    # Evaluate trigram
    print("\n" + "=" * 70)
    print("TRIGRAM EVALUATION")
    print("=" * 70)
    
    print("\nComputing perplexities...")
    train_ppl_3 = trigram_model.compute_perplexity(train_ids)
    val_ppl_3 = trigram_model.compute_perplexity(val_ids)
    test_ppl_3 = trigram_model.compute_perplexity(test_ids)
    
    print(f"\nTrigram Model Performance:")
    print(f"  Train PPL: {train_ppl_3:.2f}")
    print(f"  Val PPL:   {val_ppl_3:.2f}")
    print(f"  Test PPL:  {test_ppl_3:.2f}")
    
    # Optional: Train 5-gram model
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN 5-GRAM MODEL (n=5, alpha=1.0)")
    print("=" * 70)
    
    ngram_model_5 = BayesianNGramModel(vocab.vocab_size, n=5, alpha=1.0)
    ngram_model_5.fit(train_ids)
    
    # Evaluate 5-gram
    print("\n" + "=" * 70)
    print("5-GRAM EVALUATION")
    print("=" * 70)
    
    print("\nComputing perplexities...")
    train_ppl_5 = ngram_model_5.compute_perplexity(train_ids)
    val_ppl_5 = ngram_model_5.compute_perplexity(val_ids)
    test_ppl_5 = ngram_model_5.compute_perplexity(test_ids)
    
    print(f"\n5-gram Model Performance:")
    print(f"  Train PPL: {train_ppl_5:.2f}")
    print(f"  Val PPL:   {val_ppl_5:.2f}")
    print(f"  Test PPL:  {test_ppl_5:.2f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    random_baseline = compute_random_baseline(vocab.vocab_size)
    
    print(f"\n{'Model':<20} {'Train PPL':<12} {'Val PPL':<12} {'Test PPL':<12}")
    print("-" * 56)
    print(f"{'Random baseline':<20} {random_baseline:<12.2f} {random_baseline:<12.2f} {random_baseline:<12.2f}")
    print(f"{'Trigram':<20} {train_ppl_3:<12.2f} {val_ppl_3:<12.2f} {test_ppl_3:<12.2f}")
    print(f"{'5-gram':<20} {train_ppl_5:<12.2f} {val_ppl_5:<12.2f} {test_ppl_5:<12.2f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print(f"""
Trigram Model:
  - Improvement over random: {(1 - test_ppl_3 / random_baseline) * 100:.1f}%
  - Train-test gap: {test_ppl_3 - train_ppl_3:+.2f} PPL
  - Val-test gap: {test_ppl_3 - val_ppl_3:+.2f} PPL

5-gram Model:
  - Improvement over random: {(1 - test_ppl_5 / random_baseline) * 100:.1f}%
  - Train-test gap: {test_ppl_5 - train_ppl_5:+.2f} PPL
  - Val-test gap: {test_ppl_5 - val_ppl_5:+.2f} PPL

Interpretation:
  • PPL < {random_baseline} indicates learning above random baseline
  • Higher context length (5-gram) should reduce PPL if data is sufficient
  • Dirichlet smoothing prevents overfitting
  • Bayesian approach is interpretable: show probability for any context
""")
    
    # Sample generation
    print("\n" + "=" * 70)
    print("SAMPLE GENERATION")
    print("=" * 70)
    
    print("\nGenerating samples from trigram model (temperature=1.0):")
    
    # Use a starting context from training data
    start_context = tuple(train_ids[:2])
    
    generated = list(start_context)
    for _ in range(100):
        context = tuple(generated[-2:])
        next_token = trigram_model.sample_next_token(context)
        generated.append(next_token)
    
    generated_text = vocab.decode(generated)
    print(f"\nPrompt: '{vocab.decode(list(start_context))}'")
    print(f"Generated: {generated_text[:200]}...")
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
