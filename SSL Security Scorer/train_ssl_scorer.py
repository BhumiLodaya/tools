"""
SSL Security Score Prediction Model Training Script
Testify by Trustify - Cybersecurity Project
Uses Random Forest Regressor to calculate SSL/TLS security scores (0-100)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import warnings
import os
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_STATE = 42
N_ESTIMATORS = 100
TEST_SIZE = 0.2
MODEL_FILE = 'ssl_scoring_model.pkl'
SCALER_FILE = 'ssl_scaler.pkl'
IMPUTER_FILE = 'ssl_imputer.pkl'
FEATURE_NAMES_FILE = 'ssl_feature_names.pkl'

# Feature columns
FEATURE_COLUMNS = [
    'days_to_expiry',
    'is_self_signed',
    'cipher_bits',
    'is_revoked'
]


def generate_sample_ssl_data(n_samples=5000):
    """
    Generate sample SSL certificate data for training.
    This is used when no real data is available.
    """
    logger.info(f"Generating {n_samples} sample SSL certificate records...")
    
    np.random.seed(RANDOM_STATE)
    
    # Generate realistic SSL features
    data = {
        'days_to_expiry': np.random.choice(
            [-30, -10, 0, 30, 90, 180, 365, 730, 1095],
            size=n_samples,
            p=[0.05, 0.05, 0.05, 0.1, 0.15, 0.25, 0.25, 0.05, 0.05]
        ),
        'is_self_signed': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'cipher_bits': np.random.choice([56, 128, 192, 256], size=n_samples, p=[0.05, 0.30, 0.15, 0.50]),
        'is_revoked': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'protocol_version': np.random.choice(['SSLv3', 'TLSv1.0', 'TLSv1.1', 'TLSv1.2', 'TLSv1.3'],
                                             size=n_samples, p=[0.02, 0.05, 0.08, 0.35, 0.50]),
        'has_extended_validation': np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30]),
        'key_size': np.random.choice([1024, 2048, 4096], size=n_samples, p=[0.10, 0.70, 0.20]),
    }
    
    # Add some missing values (realistic scenario)
    mask = np.random.random(n_samples) < 0.02
    data['cipher_bits'] = data['cipher_bits'].astype(float)
    data['cipher_bits'][mask] = np.nan
    
    df = pd.DataFrame(data)
    
    logger.info("Sample SSL data generated successfully")
    return df


def calculate_security_score(df):
    """
    Calculate SSL security score (0-100) based on weighted security best practices.
    
    Scoring logic:
    - Days to expiry: 25 points (0 if expired, scaled up to 25 for valid certs)
    - Self-signed: -30 points penalty
    - Cipher bits: 20 points (scaled based on encryption strength)
    - Revoked: -50 points penalty
    - Protocol version: 20 points (higher for newer protocols)
    - Extended validation: +10 points bonus
    - Key size: 15 points (scaled based on key strength)
    
    Base score: 100 points
    """
    logger.info("Calculating security scores based on SSL best practices...")
    
    scores = np.ones(len(df)) * 100.0  # Start with perfect score
    
    # 1. Days to expiry scoring (25 points max)
    # Expired certificates get 0, valid ones scaled by time remaining
    expiry_score = df['days_to_expiry'].apply(lambda x: max(0, min(25, x / 365 * 25)))
    scores += expiry_score - 25  # Adjust from base
    
    # 2. Self-signed penalty (-30 points)
    scores -= df['is_self_signed'] * 30
    
    # 3. Cipher bits scoring (20 points max)
    cipher_score = df['cipher_bits'].fillna(0).apply(
        lambda x: 0 if x < 128 else (10 if x == 128 else (15 if x == 192 else 20))
    )
    scores += cipher_score - 20  # Adjust from base
    
    # 4. Revoked penalty (-50 points)
    scores -= df['is_revoked'] * 50
    
    # 5. Protocol version scoring (20 points max) if available
    if 'protocol_version' in df.columns:
        protocol_map = {
            'SSLv3': 0,
            'TLSv1.0': 5,
            'TLSv1.1': 10,
            'TLSv1.2': 15,
            'TLSv1.3': 20
        }
        protocol_score = df['protocol_version'].map(protocol_map).fillna(10)
        scores += protocol_score - 20  # Adjust from base
    
    # 6. Extended validation bonus (+10 points)
    if 'has_extended_validation' in df.columns:
        scores += df['has_extended_validation'] * 10 - 10  # Adjust from base
    
    # 7. Key size scoring (15 points max)
    if 'key_size' in df.columns:
        key_score = df['key_size'].apply(
            lambda x: 0 if x < 2048 else (10 if x == 2048 else 15)
        )
        scores += key_score - 15  # Adjust from base
    
    # Ensure scores are within 0-100 range
    scores = np.clip(scores, 0, 100)
    
    logger.info(f"Security scores calculated - Range: [{scores.min():.1f}, {scores.max():.1f}]")
    logger.info(f"Mean score: {scores.mean():.2f}, Median: {np.median(scores):.2f}")
    
    return scores


def load_or_generate_data(filepath='data/ssl_data.csv'):
    """Load SSL data from CSV or generate sample data if not available."""
    logger.info(f"Loading data from {filepath}...")
    
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
        else:
            logger.warning(f"File {filepath} not found. Generating sample data...")
            df = generate_sample_ssl_data(n_samples=5000)
            
            # Save generated data
            os.makedirs('data', exist_ok=True)
            df_to_save = df.copy()
            df_to_save['security_score'] = calculate_security_score(df)
            df_to_save.to_csv(filepath, index=False)
            logger.info(f"Sample data saved to {filepath}")
        
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df):
    """
    Preprocess SSL certificate data.
    
    Steps:
    1. Create security_score if it doesn't exist
    2. Select features
    3. Handle missing values with SimpleImputer
    4. Normalize features with StandardScaler
    """
    logger.info("Preprocessing data...")
    
    # Create security score if it doesn't exist
    if 'security_score' not in df.columns:
        logger.info("'security_score' column not found. Calculating...")
        df['security_score'] = calculate_security_score(df)
    else:
        logger.info("Using existing 'security_score' column")
    
    # Select features that are available
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    logger.info(f"Using features: {available_features}")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df['security_score'].values
    
    logger.info(f"Dataset: {len(X)} samples, {len(available_features)} features")
    logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Check for missing values
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Missing values:\n{missing_counts[missing_counts > 0]}")
    
    return X, y, available_features


def impute_and_scale(X_train, X_test):
    """
    Handle missing values and scale features.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, imputer, scaler)
    """
    logger.info("\nHandling missing values with SimpleImputer...")
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    logger.info("Scaling numerical features with StandardScaler...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    logger.info("Preprocessing completed")
    
    return X_train_scaled, X_test_scaled, imputer, scaler


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor for SSL security scoring.
    
    Args:
        X_train: Training features
        y_train: Training target (security scores)
        
    Returns:
        Trained RandomForestRegressor model
    """
    logger.info(f"\nTraining Random Forest Regressor...")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"n_estimators: {N_ESTIMATORS}")
    
    # Initialize Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        verbose=0
    )
    
    # Train model
    logger.info("Fitting model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance on train and test sets."""
    logger.info("\nEvaluating model performance...")
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    logger.info(f"\nTraining Set Metrics:")
    logger.info(f"  RMSE: {train_rmse:.3f}")
    logger.info(f"  MAE: {train_mae:.3f}")
    logger.info(f"  R² Score: {train_r2:.4f}")
    
    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  RMSE: {test_rmse:.3f}")
    logger.info(f"  MAE: {test_mae:.3f}")
    logger.info(f"  R² Score: {test_r2:.4f}")
    
    # Ensure predictions are in valid range
    y_test_pred_clipped = np.clip(y_test_pred, 0, 100)
    
    return y_test_pred_clipped


def save_model_and_artifacts(model, imputer, scaler, feature_names):
    """Save trained model, imputer, scaler, and feature names."""
    logger.info("\nSaving model and artifacts...")
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    logger.info(f"Model saved as '{MODEL_FILE}'")
    
    # Save imputer
    joblib.dump(imputer, IMPUTER_FILE)
    logger.info(f"Imputer saved as '{IMPUTER_FILE}'")
    
    # Save scaler
    joblib.dump(scaler, SCALER_FILE)
    logger.info(f"Scaler saved as '{SCALER_FILE}'")
    
    # Save feature names
    joblib.dump(feature_names, FEATURE_NAMES_FILE)
    logger.info(f"Feature names saved as '{FEATURE_NAMES_FILE}'")
    
    logger.info("\nAll artifacts saved successfully!")


def predict_ssl_score(features):
    """
    Predict SSL security score (0-100) for given certificate features.
    
    Args:
        features: Dictionary with SSL certificate features
        
    Returns:
        float: Security score between 0 and 100
    """
    # Load artifacts
    model = joblib.load(MODEL_FILE)
    imputer = joblib.load(IMPUTER_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    
    # Convert to DataFrame
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in correct order
    X = df[feature_names]
    
    # Impute and scale
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    # Predict
    score = model.predict(X_scaled)[0]
    
    # Ensure score is within valid range
    score = float(np.clip(score, 0, 100))
    
    return score


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("SSL Security Score Prediction Model Training")
    logger.info("Testify by Trustify - Cybersecurity Project")
    logger.info("="*70)
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    # Step 1: Load or generate data
    df = load_or_generate_data('data/ssl_data.csv')
    
    # Step 2: Preprocess data
    X, y, feature_names = preprocess_data(df)
    
    # Step 3: Split data
    logger.info(f"\nSplitting data: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Step 4: Impute and scale
    X_train_scaled, X_test_scaled, imputer, scaler = impute_and_scale(X_train, X_test)
    
    # Step 5: Train Random Forest
    model = train_random_forest(X_train_scaled, y_train)
    
    # Step 6: Evaluate model
    y_test_pred = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Step 7: Feature importance
    logger.info("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for idx, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 8: Save model and artifacts
    save_model_and_artifacts(model, imputer, scaler, feature_names)
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved as: {MODEL_FILE}")
    logger.info("="*70)
    
    # Step 9: Demonstrate usage
    logger.info("\n" + "="*70)
    logger.info("Example Usage:")
    logger.info("="*70)
    
    # Example: Strong SSL configuration
    example_strong = {
        'days_to_expiry': 180,
        'is_self_signed': 0,
        'cipher_bits': 256,
        'is_revoked': 0
    }
    
    score = predict_ssl_score(example_strong)
    logger.info(f"\nStrong SSL Configuration:")
    logger.info(f"  {example_strong}")
    logger.info(f"  Security Score: {score:.2f}/100")
    
    # Example: Weak SSL configuration
    example_weak = {
        'days_to_expiry': -10,  # Expired
        'is_self_signed': 1,
        'cipher_bits': 128,
        'is_revoked': 0
    }
    
    score = predict_ssl_score(example_weak)
    logger.info(f"\nWeak SSL Configuration (Expired + Self-Signed):")
    logger.info(f"  {example_weak}")
    logger.info(f"  Security Score: {score:.2f}/100")
    
    # Example: Revoked certificate
    example_revoked = {
        'days_to_expiry': 90,
        'is_self_signed': 0,
        'cipher_bits': 256,
        'is_revoked': 1
    }
    
    score = predict_ssl_score(example_revoked)
    logger.info(f"\nRevoked Certificate:")
    logger.info(f"  {example_revoked}")
    logger.info(f"  Security Score: {score:.2f}/100")


if __name__ == "__main__":
    main()
