# Quick Reference: 4 Architectural Changes

## Change 1: Embedding Dimension (50 → 128)
**Cell 5, Line ~182**
```python
def create_lstm_model(vocab_size, max_len, num_classes, embedding_dim=128):  # Changed from 50
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, ...)  # Now 128-dim
```

---

## Change 2: Continuous Targets for Ordinal Regression
**Cell 3, Line ~155**
```python
# Convert ranks to continuous float values for MSE loss
y_ranks_continuous = y_ranks.reshape(-1, 1).astype(np.float32)  # Changed from categorical
```

---

## Change 3: Optimized Padding with 95th Percentile
**Cell 3, Line ~135**
```python
# Use 95th percentile instead of absolute maximum
max_sequence_length = int(np.percentile(sequence_lengths, 95))  # Changed from max()
```

---

## Change 4: MSE Loss & Single Output for Regression
**Cell 5, Lines ~217-220**
```python
lstm_model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # Changed from 'categorical_crossentropy'
    metrics=['mae', 'mse']       # Changed from ['accuracy']
)
```

And:
```python
Dense(1, activation='relu')  # Changed from Dense(num_classes, activation='softmax')
```

---

## Change 5: Fixed Prediction Function (Per-Location Ranks)
**Cell 8, Lines ~377-391**
```python
# Predict different rank for each location
predictions = []
for i in range(len(location_ids)):
    seq_to_position = sequence_padded[:, :i+1]  # Context up to position i
    if seq_to_position.shape[1] < max_sequence_length:
        seq_to_position = np.pad(seq_to_position, 
                                ((0, 0), (0, max_sequence_length - seq_to_position.shape[1])), 
                                'constant', constant_values=0)
    rank_pred = lstm_model.predict(seq_to_position, verbose=0)[0][0]
    predicted_rank = int(np.round(np.clip(rank_pred, 1, len(location_ids))))
    predictions.append(predicted_rank)  # Different rank per location!
```

---

## Change 6: Updated Evaluation Metrics
**Cell 7, Lines ~295-327**
- ✅ Removed: `classification_report()`, `confusion_matrix()`, accuracy plots
- ✅ Added: Regression metrics (MSE, MAE, RMSE, R²)
- ✅ Changed: History plots to show MAE instead of accuracy
- ✅ Added: Prediction error distribution histogram

---

## Change 7: Updated Imports
**Cell 1, Lines ~14-16**
```python
# Removed: accuracy_score, classification_report, confusion_matrix, to_categorical
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

---

## Verification Checklist
- ✅ Embedding dimension: 128
- ✅ Loss function: 'mean_squared_error'
- ✅ Output layer: Dense(1, activation='relu')
- ✅ Target shape: (samples, 1) continuous values
- ✅ Padding: 95th percentile
- ✅ Metrics: mae, mse (no accuracy)
- ✅ Prediction function: Loops through each location
- ✅ No syntax errors
- ✅ Ready to train!
