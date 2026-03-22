# Model 5: TextCNN — 1D Convolutional Neural Network

## Model Architecture

Multi-kernel 1D CNN inspired by the seminal paper:  
> Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.

```
Input tokens (padded to 300 tokens)
        ↓
Embedding Layer  [vocab=20 000, dim=128]
        ↓  (B, 300, 128) → permute → (B, 128, 300)
┌─────────────────────────────────────────┐
│  Conv1d(128→128, kernel=2) + ReLU       │
│  Conv1d(128→128, kernel=3) + ReLU       │
│  Conv1d(128→128, kernel=4) + ReLU       │
│  Conv1d(128→128, kernel=5) + ReLU       │
└─────────────────────────────────────────┘
        ↓  Global Max Pooling over time for each kernel
Concatenate 4 × 128-dim vectors  →  512-dim
        ↓
Dropout (p=0.5)
        ↓
Linear → 2 (Negative / Positive)
        ↓
CrossEntropyLoss
```

| Hyperparameter | Value |
|----------------|-------|
| Vocab size | 20 000 (+ PAD, UNK) |
| Max sequence length | 300 (fixed, zero-padded) |
| Embedding dim | 128 |
| Num filters per kernel | 128 |
| Kernel sizes | 2, 3, 4, 5 |
| Dropout | 0.5 |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | StepLR (step=5, gamma=0.5) |
| Batch size | 128 |

## Techniques Applied

### 1. Multi-Kernel Convolution (N-gram Feature Detection)
Each kernel size acts like a learned n-gram detector:
- `kernel=2` → bigram features (e.g., "not good")
- `kernel=3` → trigram features (e.g., "really not bad")
- `kernel=4`, `5` → longer phrase patterns

Using **multiple kernel sizes in parallel** lets the model capture patterns at different granularities simultaneously.

### 2. Global Max Pooling
After convolution, a `max` over the time dimension extracts the most activated feature for each filter — i.e., "did this n-gram pattern appear anywhere in the review?" This makes the model invariant to position.

### 3. Fixed-length Padding
Unlike the RNN models, CNNs require a fixed sequence length. Reviews are truncated or zero-padded to 300 tokens.

### 4. High Dropout (0.5)
Heavier dropout than the RNN models because CNNs overfit more quickly on text without recurrent inductive bias.

### 5. StepLR Scheduler
Halves the learning rate every 5 epochs, providing a simple and effective decay strategy for CNNs.

## CNN vs RNN Trade-offs

| Aspect | TextCNN | BiLSTM/BiGRU |
|--------|---------|--------------|
| Training speed | **Fast** (parallelisable) | Slower (sequential) |
| Long-range dependencies | Limited | Strong |
| Local n-gram features | **Strong** | Moderate |
| Expected accuracy | ~88–91% | ~89–91% |

CNNs are usually **2–5× faster** to train than RNNs and often match them on sentiment tasks because sentiment is often determined by local phrases.

## Expected Results

| Metric | Baseline (A1) | TextCNN |
|--------|--------------|---------|
| Test Accuracy | ~77% | ~88–91% |

## How to Run

```bash
pip install torch pandas scikit-learn matplotlib seaborn
# Place 'IMDB Dataset.csv' in the notebook directory
jupyter notebook notebooks/textcnn.ipynb
```
