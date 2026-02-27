"""
Password Strength Classification Model Training Script
Testify by Trustify - Cybersecurity Project
Uses character-level LSTM for password strength prediction
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MAX_LENGTH = 30  # Maximum password length for padding
EMBEDDING_DIM = 64  # Embedding dimension
LSTM_UNITS = 128  # LSTM units
DROPOUT_RATE = 0.4  # Dropout rate
EPOCHS = 20  # Training epochs
BATCH_SIZE = 128  # Batch size
TEST_SIZE = 0.2  # 20% test data
RANDOM_STATE = 42  # For reproducibility


def load_data(filepath='data/data.csv'):
    """Load password dataset from CSV file."""
    logger.info(f"Loading data from {filepath}...")
    # Handle malformed rows with extra fields by skipping them
    df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def preprocess_data(df):
    """Preprocess password data - handle missing values and prepare for tokenization."""
    logger.info("Preprocessing data...")
    
    # Handle missing values in password column
    initial_count = len(df)
    df = df.dropna(subset=['password'])
    df = df[df['password'].astype(str).str.strip() != '']
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} records with missing/empty passwords")
    
    # Convert passwords to strings (in case of mixed types)
    df['password'] = df['password'].astype(str)
    
    # Check strength distribution
    strength_dist = df['strength'].value_counts().sort_index()
    logger.info(f"Strength distribution:\n{strength_dist}")
    
    return df


def create_character_tokenizer(passwords):
    """Create character-level tokenizer for passwords."""
    logger.info("Creating character-level tokenizer...")
    
    # Convert passwords to character sequences (space-separated characters)
    char_sequences = [' '.join(list(password)) for password in passwords]
    
    # Create tokenizer at character level
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(passwords)
    
    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"Vocabulary size (unique characters): {vocab_size}")
    logger.info(f"Sample characters: {list(tokenizer.word_index.keys())[:20]}")
    
    return tokenizer, vocab_size


def prepare_sequences(tokenizer, passwords, max_length=MAX_LENGTH):
    """Convert passwords to padded sequences."""
    logger.info(f"Converting passwords to sequences and padding to length {max_length}...")
    
    # Convert passwords to sequences
    sequences = tokenizer.texts_to_sequences(passwords)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    logger.info(f"Sequences shape: {padded_sequences.shape}")
    
    return padded_sequences


def build_model(vocab_size, max_length=MAX_LENGTH, num_classes=3):
    """Build Bidirectional LSTM model for password strength classification."""
    logger.info("Building model architecture...")
    
    model = Sequential([
        # Embedding layer
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            input_length=max_length,
            name='embedding_layer'
        ),
        
        # Bidirectional LSTM layer (captures patterns from both directions)
        Bidirectional(
            LSTM(LSTM_UNITS, return_sequences=False),
            name='bidirectional_lstm'
        ),
        
        # Dropout for regularization
        Dropout(DROPOUT_RATE, name='dropout_1'),
        
        # Dense hidden layer
        Dense(64, activation='relu', name='dense_hidden'),
        
        # Dropout for regularization
        Dropout(DROPOUT_RATE / 2, name='dropout_2'),
        
        # Output layer with softmax for 3 classes
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("\nModel Architecture:")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with validation."""
    logger.info("\nStarting model training...")
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    logger.info("\nEvaluating model on test set...")
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get predictions for additional metrics
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Per-class accuracy
    logger.info("\nPer-class performance:")
    for class_idx in range(3):
        mask = y_test == class_idx
        if mask.sum() > 0:
            class_accuracy = (y_pred_classes[mask] == class_idx).mean()
            logger.info(f"  Class {class_idx}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    return test_accuracy


def save_model_and_tokenizer(model, tokenizer):
    """Save trained model and tokenizer for later use."""
    logger.info("\nSaving model and tokenizer...")
    
    # Save model in HDF5 format
    model.save('password_strength_model.h5')
    logger.info("Model saved as 'password_strength_model.h5'")
    
    # Save tokenizer using pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info("Tokenizer saved as 'tokenizer.pkl'")
    
    logger.info("\nModel and tokenizer saved successfully!")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Password Strength Classification Model Training")
    logger.info("Testify by Trustify - Cybersecurity Project")
    logger.info("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Step 1: Load data
    df = load_data('data/data.csv')
    
    # Step 2: Preprocess data
    df = preprocess_data(df)
    
    # Step 3: Prepare features and labels
    passwords = df['password'].values
    strengths = df['strength'].values
    
    # Step 4: Create character-level tokenizer
    tokenizer, vocab_size = create_character_tokenizer(passwords)
    
    # Step 5: Prepare sequences
    X = prepare_sequences(tokenizer, passwords, MAX_LENGTH)
    y = strengths
    
    # Step 6: Split data into train and test sets (80/20 split)
    logger.info(f"\nSplitting data: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Step 7: Build model
    model = build_model(vocab_size, MAX_LENGTH, num_classes=3)
    
    # Step 8: Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Step 9: Evaluate model
    final_accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 10: Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer)
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Final Test Accuracy: {final_accuracy*100:.2f}%")
    logger.info("="*60)


if __name__ == "__main__":
    main()
