"""
DNS Security Health Score Model Training Script
Testify by Trustify - Cybersecurity Project
Uses Random Forest Regressor to generate DNS Security Health Scores (0-100)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import warnings
import os
import math
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
MODEL_FILE = 'dns_health_model.pkl'
SCALER_FILE = 'dns_scaler.pkl'
FEATURE_NAMES_FILE = 'dns_feature_names.pkl'


def calculate_domain_entropy(domain):
    """Calculate Shannon entropy of a domain name."""
    if not domain or len(domain) == 0:
        return 0.0
    
    entropy = 0
    for char in set(domain):
        p = domain.count(char) / len(domain)
        entropy -= p * math.log2(p)
    return entropy


def generate_sample_dns_data(n_samples=3000):
    """
    Generate sample DNS data with security features.
    This is used when no real data is available.
    """
    logger.info(f"Generating {n_samples} sample DNS records...")
    
    np.random.seed(RANDOM_STATE)
    
    # Generate realistic domain features
    data = {
        'domain_length': np.random.randint(5, 50, n_samples),
        'entropy': np.random.uniform(1.5, 4.5, n_samples),
        'num_subdomains': np.random.randint(0, 5, n_samples),
        'has_dnssec': np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
        'has_ds_records': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'has_spf': np.random.choice([0, 1], n_samples, p=[0.50, 0.50]),
        'has_dmarc': np.random.choice([0, 1], n_samples, p=[0.60, 0.40]),
        'record_types_count': np.random.randint(1, 8, n_samples),
        'ttl_value': np.random.choice([300, 3600, 86400], n_samples),
        'is_malicious': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'num_digits': np.random.randint(0, 10, n_samples),
        'num_hyphens': np.random.randint(0, 3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    logger.info("Sample DNS data generated successfully")
    return df


def calculate_health_score(df):
    """
    Calculate DNS Security Health Score (0-100) based on security best practices.
    
    Scoring logic (as specified):
    - Start at 50 points (base score)
    - Add 20 points if DNSSEC or DS records are present
    - Add 15 points if SPF or DMARC (TXT records) are present
    - Subtract 30 points if domain is labeled malicious
    - Additional factors: domain characteristics, record diversity
    """
    logger.info("Calculating DNS Health Scores based on security best practices...")
    
    scores = np.ones(len(df)) * 50.0  # Start at 50 points
    
    # 1. DNSSEC / DS Records (+20 points)
    dnssec_bonus = (df.get('has_dnssec', 0) | df.get('has_ds_records', 0)) * 20
    scores += dnssec_bonus
    
    # 2. SPF / DMARC in TXT records (+15 points)
    email_security_bonus = (df.get('has_spf', 0) | df.get('has_dmarc', 0)) * 15
    scores += email_security_bonus
    
    # 3. Malicious domain penalty (-30 points)
    malicious_penalty = df.get('is_malicious', 0) * 30
    scores -= malicious_penalty
    
    # 4. Additional scoring factors
    
    # Domain length (reasonable length is better)
    if 'domain_length' in df.columns:
        # Optimal length 10-30 characters
        length_score = df['domain_length'].apply(
            lambda x: 5 if 10 <= x <= 30 else 0
        )
        scores += length_score
    
    # Entropy (higher entropy is more random/suspicious for legitimate domains)
    if 'entropy' in df.columns:
        # Lower entropy (2.0-3.0) is better for legitimate domains
        entropy_score = df['entropy'].apply(
            lambda x: 5 if 2.0 <= x <= 3.0 else (0 if x > 3.5 else 2)
        )
        scores += entropy_score
    
    # Record diversity (more record types = more complete setup)
    if 'record_types_count' in df.columns:
        record_score = df['record_types_count'].apply(
            lambda x: min(10, x * 2)  # Up to 10 points for 5+ record types
        )
        scores += record_score
    
    # TTL value (higher TTL = more stable)
    if 'ttl_value' in df.columns:
        ttl_score = df['ttl_value'].apply(
            lambda x: 5 if x >= 3600 else 2
        )
        scores += ttl_score
    
    # Subdomain structure (fewer subdomains is simpler/cleaner)
    if 'num_subdomains' in df.columns:
        subdomain_score = df['num_subdomains'].apply(
            lambda x: 5 if x <= 2 else 0
        )
        scores += subdomain_score
    
    # Ensure scores are within 0-100 range
    scores = np.clip(scores, 0, 100)
    
    logger.info(f"Health scores calculated - Range: [{scores.min():.1f}, {scores.max():.1f}]")
    logger.info(f"Mean score: {scores.mean():.2f}, Median: {np.median(scores):.2f}")
    
    return scores


def load_or_generate_data(filepath='dns_data.csv'):
    """Load DNS data from CSV or generate sample data if not available."""
    logger.info(f"Loading data from {filepath}...")
    
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
        else:
            logger.warning(f"File {filepath} not found. Generating sample data...")
            df = generate_sample_dns_data(n_samples=3000)
            
            # Save generated data
            df_to_save = df.copy()
            df_to_save['health_score'] = calculate_health_score(df)
            df_to_save.to_csv(filepath, index=False)
            logger.info(f"Sample data saved to {filepath}")
        
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df):
    """
    Preprocess DNS data.
    
    Steps:
    1. Create health_score if it doesn't exist
    2. Select features
    3. Handle missing values
    """
    logger.info("Preprocessing data...")
    
    # Create health score if it doesn't exist
    if 'health_score' not in df.columns:
        logger.info("'health_score' column not found. Calculating...")
        df['health_score'] = calculate_health_score(df)
    else:
        logger.info("Using existing 'health_score' column")
    
    # Define feature columns (prioritize specified features)
    primary_features = ['domain_length', 'entropy', 'record_types_count']
    
    # Add additional features if available
    additional_features = [
        'num_subdomains', 'has_dnssec', 'has_ds_records', 
        'has_spf', 'has_dmarc', 'ttl_value',
        'num_digits', 'num_hyphens', 'num_dots',
        'num_vowels', 'digit_ratio', 'letter_ratio',
        'vowel_ratio', 'repeating_char_ratio',
        'alpha_numeric_ratio', 'special_char_ratio'
    ]
    
    # Select available features
    available_features = []
    for feat in primary_features + additional_features:
        if feat in df.columns:
            available_features.append(feat)
    
    if not available_features:
        logger.warning("No features found in expected columns. Using all numerical columns.")
        # Exclude non-feature columns
        exclude_cols = ['health_score', 'domain', 'label', 'Label', 'is_malicious']
        available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col not in exclude_cols and col != 'health_score']
    
    logger.info(f"Using {len(available_features)} features: {available_features[:10]}...")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df['health_score'].values
    
    # Handle missing values
    X = X.fillna(X.median())
    
    logger.info(f"Dataset: {len(X)} samples, {len(available_features)} features")
    logger.info(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    return X, y, available_features


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor for DNS health scoring.
    
    Args:
        X_train: Training features
        y_train: Training target (health scores)
        
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
        max_depth=20,
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
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    logger.info(f"\nTraining Set Metrics:")
    logger.info(f"  RMSE: {train_rmse:.3f}")
    logger.info(f"  MAE: {train_mae:.3f}")
    logger.info(f"  R² Score: {train_r2:.4f}")
    
    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  RMSE: {test_rmse:.3f}")
    logger.info(f"  MAE: {test_mae:.3f}")
    logger.info(f"  R² Score: {test_r2:.4f}")
    
    return y_test_pred


def save_model_and_artifacts(model, scaler, feature_names):
    """Save trained model, scaler, and feature names."""
    logger.info("\nSaving model and artifacts...")
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    logger.info(f"Model saved as '{MODEL_FILE}'")
    
    # Save scaler
    joblib.dump(scaler, SCALER_FILE)
    logger.info(f"Scaler saved as '{SCALER_FILE}'")
    
    # Save feature names
    joblib.dump(feature_names, FEATURE_NAMES_FILE)
    logger.info(f"Feature names saved as '{FEATURE_NAMES_FILE}'")
    
    logger.info("\nAll artifacts saved successfully!")


def get_trustify_score(domain_features):
    """
    Get DNS Security Health Score for React frontend integration.
    Returns a clean integer from 0-100.
    
    Args:
        domain_features: Dictionary with DNS domain features
        
    Returns:
        int: Security health score between 0 and 100
    """
    # Load artifacts
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    
    # Convert to DataFrame
    if isinstance(domain_features, dict):
        df = pd.DataFrame([domain_features])
    else:
        df = domain_features
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in correct order
    X = df[feature_names]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    score = model.predict(X_scaled)[0]
    
    # Ensure score is within valid range and return as integer
    score = int(np.clip(round(score), 0, 100))
    
    return score


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("DNS Security Health Score Model Training")
    logger.info("Testify by Trustify - Cybersecurity Project")
    logger.info("="*70)
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    # Step 1: Load or generate data
    df = load_or_generate_data('dns_data.csv')
    
    # Step 2: Preprocess data
    X, y, feature_names = preprocess_data(df)
    
    # Step 3: Split data
    logger.info(f"\nSplitting data: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Step 4: Scale features
    logger.info("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Train Random Forest
    model = train_random_forest(X_train_scaled, y_train)
    
    # Step 6: Evaluate model
    y_test_pred = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Step 7: Feature importance
    logger.info("\nTop 10 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    for idx, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 8: Save model and artifacts
    save_model_and_artifacts(model, scaler, feature_names)
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved as: {MODEL_FILE}")
    logger.info("="*70)
    
    # Step 9: Demonstrate usage
    logger.info("\n" + "="*70)
    logger.info("Example Usage for React Frontend:")
    logger.info("="*70)
    
    # Example: Strong DNS configuration
    example_strong = {
        'domain_length': 15,
        'entropy': 2.5,
        'record_types_count': 6,
        'num_subdomains': 1,
        'has_dnssec': 1,
        'has_ds_records': 1,
        'has_spf': 1,
        'has_dmarc': 1,
        'ttl_value': 3600,
        'num_digits': 0,
        'num_hyphens': 0
    }
    
    score = get_trustify_score(example_strong)
    logger.info(f"\nStrong DNS Configuration:")
    logger.info(f"  Features: DNSSEC ✓, SPF ✓, DMARC ✓")
    logger.info(f"  Trustify Score: {score}/100")
    
    # Example: Weak DNS configuration
    example_weak = {
        'domain_length': 45,
        'entropy': 4.2,
        'record_types_count': 2,
        'num_subdomains': 4,
        'has_dnssec': 0,
        'has_ds_records': 0,
        'has_spf': 0,
        'has_dmarc': 0,
        'ttl_value': 300,
        'num_digits': 8,
        'num_hyphens': 3
    }
    
    score = get_trustify_score(example_weak)
    logger.info(f"\nWeak DNS Configuration:")
    logger.info(f"  Features: No DNSSEC, No SPF/DMARC")
    logger.info(f"  Trustify Score: {score}/100")
    
    # Example: Malicious domain
    example_malicious = {
        'domain_length': 35,
        'entropy': 3.8,
        'record_types_count': 1,
        'num_subdomains': 0,
        'has_dnssec': 0,
        'has_ds_records': 0,
        'has_spf': 0,
        'has_dmarc': 0,
        'ttl_value': 300,
        'num_digits': 10,
        'num_hyphens': 2
    }
    
    score = get_trustify_score(example_malicious)
    logger.info(f"\nSuspicious Domain Pattern:")
    logger.info(f"  Features: High entropy, many digits")
    logger.info(f"  Trustify Score: {score}/100")
    
    logger.info("\n" + "="*70)
    logger.info("React Frontend Integration:")
    logger.info("="*70)
    logger.info("""
// JavaScript example:
const response = await fetch('/api/dns-health', {
    method: 'POST',
    body: JSON.stringify({
        domain_length: 15,
        entropy: 2.5,
        record_types_count: 6,
        has_dnssec: 1,
        has_spf: 1,
        has_dmarc: 1
    })
});
const { score } = await response.json();
console.log(`DNS Health Score: ${score}/100`);
    """)


if __name__ == "__main__":
    main()
