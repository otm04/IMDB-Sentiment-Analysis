# Model 2: Improved MLP

## Model Architecture

A deeper, regularised Multi-Layer Perceptron built on top of the Assignment 1 baseline.

| Component | Details |
|-----------|---------|
| Input features | 9 (extended from 2 in baseline) |
| Hidden layers | 3 (64 → 32 → 16 neurons) |
| Activations | ReLU |
| Regularisation | BatchNorm1d + Dropout (p=0.3) per hidden layer |
| Output | Linear → BCEWithLogitsLoss |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping | 15 epochs no improvement on val accuracy |
| Batch size | 256 |

## Techniques Applied

### 1. Extended Feature Engineering
The baseline used only 2 features (VADER compound + TextBlob polarity). This model adds 7 more hand-crafted signals:

| Feature | Description |
|---------|-------------|
| `vader_compound` | Overall VADER polarity score |
| `vader_pos` | VADER positive score |
| `vader_neg` | VADER negative score |
| `vader_neu` | VADER neutral score |
| `textblob_polar` | TextBlob sentence polarity |
| `textblob_subj` | TextBlob subjectivity score |
| `review_length` | Token count (longer reviews tend to be more opinionated) |
| `exclaim_count` | Number of `!` characters (excitement / anger) |
| `question_count` | Number of `?` characters |

All features are standardised with `StandardScaler`.

### 2. Deeper Architecture
Baseline had 1 hidden layer (8 neurons). This model uses 3 hidden layers (64 → 32 → 16), increasing representational capacity.

### 3. Batch Normalisation
Applied after each linear layer to stabilise training and allow higher learning rates.

### 4. Dropout Regularisation
`p=0.3` dropout after each hidden block reduces overfitting.

### 5. Advanced Optimisation
- **Adam** optimiser replaces SGD; adapts per-parameter learning rates.
- **Weight decay** (L2) on parameters as additional regularisation.
- **ReduceLROnPlateau** scheduler automatically reduces LR when validation loss stagnates.

### 6. Early Stopping
Monitors validation accuracy; restores the best checkpoint if no improvement for 15 epochs.

## Expected Results

| Metric | Baseline (A1) | Improved MLP |
|--------|--------------|--------------|
| Test Accuracy | ~77% | ~82–84% |

## How to Run

```bash
# Place 'IMDB Dataset.csv' in the same directory as the notebook
pip install torch pandas scikit-learn vaderSentiment textblob matplotlib seaborn
jupyter notebook notebooks/improved_mlp.ipynb
```
