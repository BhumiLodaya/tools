"""
Hash Algorithm Identifier - Training Script
Testify by Trustify - Cybersecurity Project

Generates synthetic hash data and trains a Random Forest Classifier to identify
hash algorithms (MD5, SHA-1, SHA-256) from hash strings.
"""

import hashlib
import random
import string
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_random_string(min_length=5, max_length=100):
    """Generate a random string of variable length."""
    length = random.randint(min_length, max_length)
    characters = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choice(characters) for _ in range(length))


def calculate_entropy(text):
    """
    Calculate Shannon Entropy of a string.
    Higher entropy means more randomness.
    """
    if not text:
        return 0.0
    
    # Calculate frequency of each character
    frequencies = {}
    for char in text:
        frequencies[char] = frequencies.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    text_length = len(text)
    
    for count in frequencies.values():
        probability = count / text_length
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def extract_features(hash_string):
    """
    Extract features from a hash string for classification.
    
    Features:
    - Length of hash
    - Number of digits
    - Number of letters
    - Number of alphabetic characters (a-f for hex)
    - Shannon Entropy
    - Ratio of digits to total length
    - Ratio of letters to total length
    """
    hash_length = len(hash_string)
    num_digits = sum(c.isdigit() for c in hash_string)
    num_letters = sum(c.isalpha() for c in hash_string)
    num_lowercase = sum(c.islower() for c in hash_string)
    num_uppercase = sum(c.isupper() for c in hash_string)
    
    # Count hex-specific characters (a-f)
    num_hex_letters = sum(c in 'abcdefABCDEF' for c in hash_string)
    
    # Calculate entropy
    entropy = calculate_entropy(hash_string)
    
    # Calculate ratios
    digit_ratio = num_digits / hash_length if hash_length > 0 else 0
    letter_ratio = num_letters / hash_length if hash_length > 0 else 0
    hex_letter_ratio = num_hex_letters / hash_length if hash_length > 0 else 0
    
    return {
        'hash_length': hash_length,
        'num_digits': num_digits,
        'num_letters': num_letters,
        'num_lowercase': num_lowercase,
        'num_uppercase': num_uppercase,
        'num_hex_letters': num_hex_letters,
        'entropy': entropy,
        'digit_ratio': digit_ratio,
        'letter_ratio': letter_ratio,
        'hex_letter_ratio': hex_letter_ratio
    }


def generate_synthetic_dataset(num_samples=10000):
    """
    Generate synthetic dataset with hash strings.
    
    Args:
        num_samples: Number of random strings to generate
        
    Returns:
        DataFrame with hash strings and their labels
    """
    logger.info(f"Generating {num_samples} random strings...")
    
    data = []
    
    for i in range(num_samples):
        if (i + 1) % 2000 == 0:
            logger.info(f"  Generated {i + 1}/{num_samples} strings...")
        
        # Generate random string
        random_string = generate_random_string()
        random_bytes = random_string.encode('utf-8')
        
        # Generate MD5 hash
        md5_hash = hashlib.md5(random_bytes).hexdigest()
        data.append({
            'hash_string': md5_hash,
            'algorithm': 'MD5',
            'original_length': len(random_string)
        })
        
        # Generate SHA-1 hash
        sha1_hash = hashlib.sha1(random_bytes).hexdigest()
        data.append({
            'hash_string': sha1_hash,
            'algorithm': 'SHA1',
            'original_length': len(random_string)
        })
        
        # Generate SHA-256 hash
        sha256_hash = hashlib.sha256(random_bytes).hexdigest()
        data.append({
            'hash_string': sha256_hash,
            'algorithm': 'SHA256',
            'original_length': len(random_string)
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} hash samples")
    logger.info(f"  MD5: {len(df[df['algorithm'] == 'MD5'])}")
    logger.info(f"  SHA1: {len(df[df['algorithm'] == 'SHA1'])}")
    logger.info(f"  SHA256: {len(df[df['algorithm'] == 'SHA256'])}")
    
    return df


def prepare_features(df):
    """Extract features from hash strings in the dataset."""
    logger.info("Extracting features from hash strings...")
    
    features_list = []
    for idx, hash_string in enumerate(df['hash_string']):
        if (idx + 1) % 5000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} hashes...")
        
        features = extract_features(hash_string)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Combine with original dataframe
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    logger.info(f"Feature extraction complete. Shape: {features_df.shape}")
    logger.info(f"Features: {list(features_df.columns)}")
    
    return result_df


def train_model(df):
    """
    Train Random Forest Classifier to identify hash algorithms.
    
    Args:
        df: DataFrame with features and labels
        
    Returns:
        Trained model, feature columns, and evaluation metrics
    """
    logger.info("\n" + "="*70)
    logger.info("Training Random Forest Classifier")
    logger.info("="*70)
    
    # Define feature columns
    feature_columns = [
        'hash_length',
        'num_digits',
        'num_letters',
        'num_lowercase',
        'num_uppercase',
        'num_hex_letters',
        'entropy',
        'digit_ratio',
        'letter_ratio',
        'hex_letter_ratio'
    ]
    
    X = df[feature_columns]
    y = df['algorithm']
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Feature columns: {feature_columns}")
    logger.info(f"Target classes: {y.unique()}")
    
    # Split data 80/20
    logger.info("\nSplitting data: 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Train Random Forest Classifier
    logger.info("\nTraining Random Forest Classifier...")
    logger.info("Parameters:")
    logger.info("  n_estimators: 100")
    logger.info("  max_depth: 20")
    logger.info("  min_samples_split: 5")
    logger.info("  random_state: 42")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed!")
    
    # Evaluate model
    logger.info("\n" + "="*70)
    logger.info("Model Evaluation")
    logger.info("="*70)
    
    # Training accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    logger.info(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Test accuracy
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    logger.info("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_predictions))
    
    # Confusion matrix
    logger.info("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_predictions)
    logger.info(f"\n{cm}")
    
    # Feature importance
    logger.info("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, feature_columns, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'classification_report': classification_report(y_test, test_predictions, output_dict=True),
        'confusion_matrix': cm.tolist()
    }


def save_model(model, feature_columns, metrics):
    """Save the trained model and metadata."""
    logger.info("\n" + "="*70)
    logger.info("Saving Model and Artifacts")
    logger.info("="*70)
    
    # Save model
    model_filename = 'hash_id_model.pkl'
    joblib.dump(model, model_filename)
    logger.info(f"Model saved as '{model_filename}'")
    
    # Save feature columns
    features_filename = 'hash_id_features.pkl'
    joblib.dump(feature_columns, features_filename)
    logger.info(f"Feature columns saved as '{features_filename}'")
    
    # Save metrics
    metrics_filename = 'hash_id_metrics.pkl'
    joblib.dump(metrics, metrics_filename)
    logger.info(f"Metrics saved as '{metrics_filename}'")
    
    logger.info("\nAll artifacts saved successfully!")


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("Hash Algorithm Identifier - Model Training")
    logger.info("Testify by Trustify - Cybersecurity Project")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    # Generate synthetic dataset
    df = generate_synthetic_dataset(num_samples=10000)
    
    # Save raw dataset
    dataset_filename = 'hash_dataset.csv'
    df.to_csv(dataset_filename, index=False)
    logger.info(f"\nRaw dataset saved as '{dataset_filename}'")
    
    # Extract features
    df_with_features = prepare_features(df)
    
    # Save dataset with features
    features_dataset_filename = 'hash_dataset_with_features.csv'
    df_with_features.to_csv(features_dataset_filename, index=False)
    logger.info(f"Dataset with features saved as '{features_dataset_filename}'")
    
    # Train model
    model, feature_columns, metrics = train_model(df_with_features)
    
    # Save model
    save_model(model, feature_columns, metrics)
    
    # Training summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*70)
    logger.info("Training Summary")
    logger.info("="*70)
    logger.info(f"Total samples: {len(df_with_features)}")
    logger.info(f"Training accuracy: {metrics['train_accuracy']*100:.2f}%")
    logger.info(f"Test accuracy: {metrics['test_accuracy']*100:.2f}%")
    logger.info(f"Training time: {duration:.2f} seconds")
    logger.info(f"Model saved: hash_id_model.pkl")
    logger.info("="*70)
    
    # Demo prediction
    logger.info("\n" + "="*70)
    logger.info("Demo Predictions")
    logger.info("="*70)
    
    # Test with known hashes
    test_hashes = [
        ('5d41402abc4b2a76b9719d911017c592', 'MD5'),  # MD5 of "hello"
        ('aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d', 'SHA1'),  # SHA1 of "hello"
        ('2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824', 'SHA256'),  # SHA256 of "hello"
    ]
    
    for hash_string, expected in test_hashes:
        features = extract_features(hash_string)
        features_df = pd.DataFrame([features])
        features_df = features_df[feature_columns]
        
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = max(probabilities) * 100
        
        logger.info(f"\nHash: {hash_string}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Predicted: {prediction}")
        logger.info(f"Confidence: {confidence:.2f}%")
        logger.info(f"Match: {'✓' if prediction == expected else '✗'}")
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
