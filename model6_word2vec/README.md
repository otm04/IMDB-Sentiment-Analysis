# Model 6: BiGRU with Pretrained Word2Vec Embeddings

## Model Architecture

A Bidirectional GRU whose embedding layer is **initialised from Word2Vec vectors** trained on the IMDB corpus itself using the Gensim library.

```
Step 1 – Word2Vec Training (Gensim, skip-gram, dim=128, window=5)
        ↓
Step 2 – Build embedding matrix from W2V vectors
        ↓
Input tokens (sequence)
        ↓
Embedding Layer  [from_pretrained=True, fine-tuned during training]
        ↓
BiGRU  [2 layers, hidden=128, bidirectional → 256]
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
| W2V algorithm | Skip-gram (`sg=1`) |
| W2V vector size | 128 |
| W2V window | 5 |
| W2V min count | 2 |
| W2V epochs | 10 |
| Max sequence length | 256 |
| GRU hidden dim | 128 (256 after bidirectional) |
| GRU layers | 2 |
| Dropout | 0.3 |
| Freeze embeddings | False (fine-tuned) |
| Optimiser | Adam (lr=1e-3) |
| LR Scheduler | CosineAnnealingLR (T_max=20) |
| Gradient clipping | max norm = 1.0 |
| Batch size | 64 |

## Techniques Applied

### 1. Word2Vec with Gensim (Skip-gram)
Skip-gram Word2Vec learns vector representations by predicting surrounding context words. Words that appear in similar contexts (e.g., "great" and "excellent") end up close together in vector space. This provides a semantically rich initialisation for the embedding layer.

**Why train on IMDB vs using generic pretrained vectors?**  
Domain-specific vocabulary matters. Reviews use film-specific language ("cinematography", "screenplay", "Oscar-worthy") that may be rare in general-purpose corpora like Google News.

### 2. Pretrained Embedding Initialisation
`nn.Embedding.from_pretrained(embed_matrix, freeze=False)` initialises the embedding weights from the W2V matrix. Setting `freeze=False` allows further fine-tuning during classifier training, combining prior semantic knowledge with task-specific optimisation.

### 3. Semantic Sanity Check
After training W2V, a `most_similar('great')` call verifies that neighbours are semantically sensible (e.g., "fantastic", "wonderful") before proceeding to classifier training.

### 4. All BiGRU techniques also apply
Bidirectionality, gradient clipping, cosine annealing LR, early stopping, variable-length batching via `pad_sequence`.

## Pretrained Embeddings vs Random Initialisation

| Aspect | Random Init | W2V Pretrained |
|--------|------------|----------------|
| Epoch 1 accuracy | Lower | **Higher** |
| Convergence speed | Slower | **Faster** |
| Final accuracy | ~89% | **~90–92%** |
| Training stability | Noisier | More stable |

The pretrained embeddings give the model a head start — it already "knows" that "terrible" is the opposite of "wonderful" before seeing a single labelled example.

## Expected Results

| Metric | Baseline (A1) | BiGRU + W2V |
|--------|--------------|-------------|
| Test Accuracy | ~77% | ~90–92% |

## How to Run

```bash
pip install torch pandas scikit-learn gensim matplotlib seaborn
# Place 'IMDB Dataset.csv' in the notebook directory
jupyter notebook notebooks/bigru_word2vec.ipynb
```
