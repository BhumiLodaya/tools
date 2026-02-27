# Password Strength Classification Model

**Testify by Trustify - Cybersecurity Project**

A character-level LSTM neural network for classifying password strength into 3 categories (0, 1, 2).

## Features

- **Character-level tokenization**: Analyzes passwords at the character level for better pattern recognition
- **Bidirectional LSTM**: Captures patterns from both directions of the password
- **Dropout regularization**: Prevents overfitting
- **3-class classification**: Weak (0), Medium (1), Strong (2)

## Model Architecture

```
1. Embedding Layer (64 dimensions)
2. Bidirectional LSTM (128 units)
3. Dropout (0.4)
4. Dense Layer (64 units, ReLU)
5. Dropout (0.2)
6. Output Layer (3 units, Softmax)
```

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script:

```bash
python train_password_model.py
```

The script will:
1. Load password data from `data/data.csv`
2. Preprocess and tokenize at character level
3. Split into 80% training and 20% testing sets
4. Train the Bidirectional LSTM model
5. Evaluate on the test set
6. Save the trained model and tokenizer

### Output Files

After training, you'll get:
- `password_strength_model.h5` - Trained Keras model
- `tokenizer.pkl` - Character-level tokenizer (for inference)

### Using the Model in Your Project

```python
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = keras.models.load_model('password_strength_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict password strength
def predict_password_strength(password, max_length=30):
    sequence = tokenizer.texts_to_sequences([password])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded, verbose=0)
    strength_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][strength_class]
    
    strength_labels = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
    return strength_labels[strength_class], confidence

# Example
password = "MyP@ssw0rd123!"
strength, confidence = predict_password_strength(password)
print(f"Password: {password}")
print(f"Strength: {strength} (Confidence: {confidence*100:.2f}%)")
```

## Data Format

The `data/data.csv` file should have two columns:
- `password`: The password string
- `strength`: Integer class (0, 1, or 2)

Example:
```csv
password,strength
weakpass,0
M3diumP@ss,1
V3ry$tr0ng!P@ssw0rd,2
```

## Configuration

You can modify these parameters in the script:
- `MAX_LENGTH`: Maximum password length (default: 30)
- `EMBEDDING_DIM`: Embedding dimension (default: 64)
- `LSTM_UNITS`: LSTM units (default: 128)
- `DROPOUT_RATE`: Dropout rate (default: 0.4)
- `EPOCHS`: Training epochs (default: 20)
- `BATCH_SIZE`: Batch size (default: 128)

## License

Part of the Testify by Trustify cybersecurity project.
