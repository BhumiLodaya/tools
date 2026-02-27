"""
Port Scan Anomaly Detection Model Training Script
Testify by Trustify - Cybersecurity Project
Uses Isolation Forest to detect port scanning activities and anomalous network traffic
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_STATE = 42
CONTAMINATION = 0.1  # Assuming 10% of traffic might be scanning/anomalous
MODEL_FILE = 'port_scan_model.pkl'
SCALER_FILE = 'port_scaler.pkl'
FEATURE_NAMES_FILE = 'port_feature_names.pkl'

# Features to use for port scan detection
SELECTED_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Length of Fwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Packet Length Mean',
    'Average Packet Size',
    'FIN Flag Count',
    'PSH Flag Count',
    'ACK Flag Count'
]


def load_data(filepath='data/port_scan_data.csv', sample_size=None):
    """
    Load port scan dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        sample_size: Number of rows to sample (None for all data)
    """
    logger.info(f"Loading data from {filepath}...")
    
    try:
        # Load data
        if sample_size:
            # Read in chunks and sample if file is large
            logger.info(f"Sampling {sample_size} rows from dataset...")
            df = pd.read_csv(filepath, nrows=sample_size)
        else:
            df = pd.read_csv(filepath)
        
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Columns: {len(df.columns)} features")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df, feature_columns=SELECTED_FEATURES):
    """
    Preprocess network traffic data.
    
    Steps:
    1. Select relevant features
    2. Handle inf and NaN values
    3. Remove any remaining invalid data
    """
    logger.info("Preprocessing data...")
    
    # Check which features are available
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    logger.info(f"Using {len(available_features)} features: {available_features[:5]}...")
    
    # Select available features
    df_features = df[available_features].copy()
    
    initial_count = len(df_features)
    
    # Handle infinite values - replace with NaN first
    logger.info("Handling infinite values...")
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    # Count NaN values
    nan_counts = df_features.isna().sum()
    if nan_counts.sum() > 0:
        logger.info(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
    
    # Fill NaN values with median (more robust than mean for network data)
    logger.info("Filling NaN values with column medians...")
    for col in df_features.columns:
        if df_features[col].isna().sum() > 0:
            median_val = df_features[col].median()
            if np.isnan(median_val):
                # If median is NaN, use 0
                df_features[col].fillna(0, inplace=True)
            else:
                df_features[col].fillna(median_val, inplace=True)
    
    # Remove any remaining rows with NaN
    df_features = df_features.dropna()
    
    removed_count = initial_count - len(df_features)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with invalid values")
    
    logger.info(f"Final dataset: {len(df_features)} records with {len(df_features.columns)} features")
    
    # Display statistics
    logger.info("\nFeature Statistics:")
    logger.info(df_features.describe())
    
    return df_features, available_features


def normalize_features(X, scaler=None):
    """
    Normalize features using StandardScaler.
    Especially important for port numbers and durations which have different scales.
    """
    logger.info("Normalizing features with StandardScaler...")
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Scaler fitted and features normalized")
    else:
        X_scaled = scaler.transform(X)
        logger.info("Features normalized using existing scaler")
    
    return X_scaled, scaler


def train_isolation_forest(X_train):
    """
    Train Isolation Forest for port scan anomaly detection.
    
    The model flags anomalous patterns like:
    - Sequential port scanning
    - High packet rates to single destination
    - Unusual flow patterns
    
    Returns -1 for anomalies, 1 for normal traffic.
    """
    logger.info("\nTraining Isolation Forest model...")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Contamination: {CONTAMINATION} (expecting {CONTAMINATION*100}% anomalies)")
    
    # Initialize Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        verbose=0
    )
    
    # Train model
    logger.info("Fitting model...")
    model.fit(X_train)
    logger.info("Model training completed")
    
    # Get predictions on training data for statistics
    train_predictions = model.predict(X_train)
    train_scores = model.score_samples(X_train)
    
    normal_count = (train_predictions == 1).sum()
    anomaly_count = (train_predictions == -1).sum()
    
    logger.info(f"\nTraining set predictions:")
    logger.info(f"  Normal (1): {normal_count} ({normal_count/len(train_predictions)*100:.2f}%)")
    logger.info(f"  Anomalies (-1): {anomaly_count} ({anomaly_count/len(train_predictions)*100:.2f}%)")
    logger.info(f"  Anomaly score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    logger.info(f"  Mean score: {train_scores.mean():.4f}")
    
    return model


def save_model_and_artifacts(model, scaler, feature_names):
    """Save trained model, scaler, and feature names using joblib."""
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


def detect_port_scan(traffic_data):
    """
    Detect port scanning activity in network traffic.
    
    Args:
        traffic_data: Dictionary or DataFrame with network traffic features
        
    Returns:
        dict: Detection result with prediction and anomaly score
    """
    # Load model and artifacts
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    
    # Convert to DataFrame if dict
    if isinstance(traffic_data, dict):
        df = pd.DataFrame([traffic_data])
    else:
        df = traffic_data
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in correct order
    X = df[feature_names]
    
    # Handle inf and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Normalize
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    score = model.score_samples(X_scaled)[0]
    
    # Determine severity
    if prediction == 1:
        severity = "Normal"
        risk = "Low"
    else:
        if score < -0.3:
            severity = "Critical"
            risk = "High"
        elif score < -0.15:
            severity = "Warning"
            risk = "Medium"
        else:
            severity = "Alert"
            risk = "Low"
    
    return {
        'prediction': int(prediction),
        'is_port_scan': bool(prediction == -1),
        'anomaly_score': float(score),
        'severity': severity,
        'risk_level': risk,
        'status': 'Port Scan Detected' if prediction == -1 else 'Normal Traffic'
    }


def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("Port Scan Anomaly Detection Model Training")
    logger.info("Testify by Trustify - Cybersecurity Project")
    logger.info("="*70)
    
    # Set random seed
    np.random.seed(RANDOM_STATE)
    
    # Step 1: Load data
    # Use sample_size for large datasets to speed up training
    df = load_data('data/port_scan_data.csv', sample_size=100000)
    
    # Step 2: Check for attack labels if available
    if 'Attack Type' in df.columns:
        attack_dist = df['Attack Type'].value_counts()
        logger.info(f"\nAttack Type distribution:\n{attack_dist.head(10)}")
    elif 'Label' in df.columns:
        label_dist = df['Label'].value_counts()
        logger.info(f"\nLabel distribution:\n{label_dist}")
    
    # Step 3: Preprocess data
    X_df, feature_names = preprocess_data(df)
    
    # Step 4: Normalize features
    X_scaled, scaler = normalize_features(X_df.values)
    
    # Step 5: Train Isolation Forest
    model = train_isolation_forest(X_scaled)
    
    # Step 6: Evaluate on labeled data if available
    if 'Attack Type' in df.columns or 'Label' in df.columns:
        logger.info("\n" + "="*70)
        logger.info("Evaluating on labeled data...")
        logger.info("="*70)
        
        label_col = 'Attack Type' if 'Attack Type' in df.columns else 'Label'
        
        # Get predictions for all data
        predictions = model.predict(X_scaled)
        
        # Create evaluation DataFrame
        eval_df = pd.DataFrame({
            'true_label': df[label_col].values[:len(predictions)],
            'prediction': predictions
        })
        
        # Check how model performs on different attack types
        logger.info("\nPredictions by Attack Type:")
        for attack_type in eval_df['true_label'].unique()[:10]:
            subset = eval_df[eval_df['true_label'] == attack_type]
            anomaly_count = (subset['prediction'] == -1).sum()
            total = len(subset)
            logger.info(f"  {attack_type}: {anomaly_count}/{total} "
                       f"({anomaly_count/total*100:.1f}%) flagged as anomalies")
    
    # Step 7: Save model and artifacts
    save_model_and_artifacts(model, scaler, feature_names)
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved as: {MODEL_FILE}")
    logger.info(f"Scaler saved as: {SCALER_FILE}")
    logger.info("="*70)
    
    # Step 8: Demonstrate usage
    logger.info("\n" + "="*70)
    logger.info("Example Usage:")
    logger.info("="*70)
    
    # Example normal traffic
    example_normal = {
        'Destination Port': 80,
        'Flow Duration': 15000,
        'Total Fwd Packets': 10,
        'Total Length of Fwd Packets': 5000,
        'Flow Bytes/s': 333.33,
        'Flow Packets/s': 0.67,
        'Fwd Packets/s': 0.67,
        'Bwd Packets/s': 0.33,
        'Packet Length Mean': 500,
        'Average Packet Size': 500,
        'FIN Flag Count': 1,
        'PSH Flag Count': 2,
        'ACK Flag Count': 8
    }
    
    result = detect_port_scan(example_normal)
    logger.info(f"\nNormal Traffic Analysis:")
    logger.info(f"  Prediction: {result['prediction']} ({result['status']})")
    logger.info(f"  Anomaly Score: {result['anomaly_score']:.4f}")
    logger.info(f"  Severity: {result['severity']}")
    
    # Example port scan traffic
    example_port_scan = {
        'Destination Port': 22,
        'Flow Duration': 100,
        'Total Fwd Packets': 1,
        'Total Length of Fwd Packets': 60,
        'Flow Bytes/s': 600,
        'Flow Packets/s': 10,
        'Fwd Packets/s': 10,
        'Bwd Packets/s': 0,
        'Packet Length Mean': 60,
        'Average Packet Size': 60,
        'FIN Flag Count': 0,
        'PSH Flag Count': 0,
        'ACK Flag Count': 0
    }
    
    result = detect_port_scan(example_port_scan)
    logger.info(f"\nSuspicious Traffic Analysis:")
    logger.info(f"  Prediction: {result['prediction']} ({result['status']})")
    logger.info(f"  Anomaly Score: {result['anomaly_score']:.4f}")
    logger.info(f"  Severity: {result['severity']}")


if __name__ == "__main__":
    main()
