# Model 3: Bidirectional LSTM

## Model Architecture

A 2-layer Bidirectional LSTM that reads the raw token sequence (not hand-crafted features).

```
Input tokens (sequence)
        ↓
Embedding Layer  [vocab=20 000, dim=128]
        ↓
BiLSTM  [2 layers, hidden=128, bidirectional → 256 per step]
        ↓
Concatenate last forward & backward hidden states  → 256-dim
        ↓
Dropout (p=0.3)
        ↓
Linear → 2 (Negative / Positive)
        ↓
CrossEntropyLoss
```

| Hyperparameter | Value |
|----------------|-------|
| Vocab size | 20 000 (+ PAD, UNK) |
| Max sequence length | 256 tokens |
| Embedding dim | 128 |
| Hidden dim | 128 (256 after bidirectional concat) |
| LSTM layers | 2 |
| Dropout | 0.3 |
| Optimiser | Adam (lr=1e-3) |
| LR Scheduler | CosineAnnealingLR (T_max=20) |
| Gradient clipping | max norm = 1.0 |
| Batch size | 64 |
| Early stopping | 5 epochs patience |

## Techniques Applied

### 1. Raw Text Tokenisation
Reviews are lowercased, stripped of HTML tags and punctuation, and split into word tokens — no manual feature engineering required.

### 2. Trainable Embedding Layer
A randomly-initialised embedding matrix is learned end-to-end alongside the rest of the network.

### 3. Bidirectionality
The LSTM processes each sequence both left-to-right and right-to-left. Sentiment sometimes depends on context that appears later in a review (e.g., "not good at all"), and the backward pass captures that.

### 4. Gradient Clipping
`nn.utils.clip_grad_norm_(model.parameters(), 1.0)` prevents exploding gradients, which are common in RNNs on long sequences.

### 5. Cosine Annealing LR Schedule
Smoothly reduces the learning rate following a cosine curve, helping the model converge to a better minimum.

### 6. Variable-length Sequences with Padding
`pad_sequence` pads each mini-batch only to the longest sequence in the batch, rather than a fixed global maximum — this is more memory-efficient.

## Expected Results

| Metric | Baseline (A1) | BiLSTM |
|--------|--------------|--------|
| Test Accuracy | ~77% | ~88–90% |

## How to Run

```bash
pip install torch pandas scikit-learn matplotlib seaborn
# Place 'IMDB Dataset.csv' in the notebook directory
jupyter notebook notebooks/lstm.ipynb
```
