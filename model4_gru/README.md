# Model 4: Bidirectional GRU with Attention

## Model Architecture

A 2-layer Bidirectional GRU enhanced with an additive attention mechanism over all time steps.

```
Input tokens (sequence)
        ↓
Embedding Layer  [vocab=20 000, dim=128]
        ↓
BiGRU  [2 layers, hidden=128, bidirectional → 256 per step]
        ↓
Additive Attention over all time steps  → context vector (256-dim)
        ↓
Dropout (p=0.3)
        ↓
Linear → 2 (Negative / Positive)
        ↓
CrossEntropyLoss
```

### Attention Module (Bahdanau-style)

```
scores  = Linear(256 → 1)  applied to each time step
weights = softmax(scores)
context = Σ weights_t × hidden_t
```

The attention layer lets the model assign different importance to each word, effectively focusing on the most sentiment-bearing tokens (e.g., "brilliant", "terrible", "not").

| Hyperparameter | Value |
|----------------|-------|
| Vocab size | 20 000 (+ PAD, UNK) |
| Max sequence length | 256 tokens |
| Embedding dim | 128 |
| Hidden dim | 128 (256 after bidirectional) |
| GRU layers | 2 |
| Dropout | 0.3 |
| Optimiser | Adam (lr=1e-3) |
| LR Scheduler | CosineAnnealingLR (T_max=20) |
| Gradient clipping | max norm = 1.0 |
| Batch size | 64 |

## Techniques Applied

### 1. GRU vs LSTM
GRU uses only two gates (reset, update) instead of LSTM's three (input, forget, output). This means:
- Fewer parameters → faster training
- Less prone to overfitting on moderate-sized datasets
- Often comparable accuracy to LSTM

### 2. Attention Mechanism
Instead of using only the final hidden state (as in the LSTM model), attention aggregates **all** time steps using learned weights. This is especially effective for long reviews where important words may appear anywhere.

### 3. All techniques from BiLSTM also apply
Bidirectionality, gradient clipping, cosine annealing LR, early stopping, variable-length batching.

## GRU vs LSTM Comparison

| Aspect | BiLSTM (Model 3) | BiGRU + Attention (Model 4) |
|--------|-----------------|---------------------------|
| Gates | 3 (input, forget, output) | 2 (reset, update) |
| Parameters | More | Fewer |
| Context usage | Last hidden state | Attention over all steps |
| Training speed | Slower | Faster |
| Expected accuracy | ~88–90% | ~89–91% |

## Expected Results

| Metric | Baseline (A1) | BiGRU + Attention |
|--------|--------------|-------------------|
| Test Accuracy | ~77% | ~89–91% |

## How to Run

```bash
pip install torch pandas scikit-learn matplotlib seaborn
# Place 'IMDB Dataset.csv' in the notebook directory
jupyter notebook notebooks/gru.ipynb
```
