"""
HTTP Header Anomaly Detection Model Training Script
Testify by Trustify - Cybersecurity Project
Uses Isolation Forest to detect anomalous HTTP header patterns
"""

import numpy as np
import pandas as pd
import pickle
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
CONTAMINATION = 0.1  # Expected proportion of outliers
MODEL_FILE = 'header_analyzer_model.pkl'
SCALER_FILE = 'header_scaler.pkl'
FEATURE_COLUMNS_FILE = 'header_feature_columns.pkl'


def load_data(filepath='data/http_headers.csv'):
    """Load HTTP headers dataset from CSV file."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def engineer_features(df):
    """
    Convert HTTP headers into numerical features.
    
    Feature types:
    1. Categorical flags (1 if header present, 0 otherwise)
    2. Numerical features (header count, payload length, etc.)
    """
    logger.info("Engineering features from HTTP headers...")
    
    # Create a copy for feature engineering
    features_df = pd.DataFrame()
    
    # Define common HTTP headers to check for presence
    header_columns = [
        'Method', 'User-Agent', 'Pragma', 'Cache-Control', 'Accept',
        'Accept-encoding', 'Accept-charset', 'language', 'host', 'cookie',
        'content-type', 'connection'
    ]
    
    # 1. Categorical Features - Binary flags for header presence
    for header in header_columns:
        if header in df.columns:
            # 1 if header exists and is not empty/null, 0 otherwise
            features_df[f'has_{header.lower().replace("-", "_")}'] = (
                df[header].notna() & (df[header].astype(str).str.strip() != '')
            ).astype(int)
    
    # 2. Method type flags (GET, POST, etc.)
    if 'Method' in df.columns:
        for method in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']:
            features_df[f'method_{method.lower()}'] = (
                df['Method'].str.upper() == method
            ).astype(int)
    
    # 3. Numerical Features
    
    # Count of headers present
    header_count = 0
    for header in header_columns:
        if header in df.columns:
            header_count += (df[header].notna() & 
                           (df[header].astype(str).str.strip() != '')).astype(int)
    features_df['header_count'] = header_count
    
    # Payload/Content length
    if 'lenght' in df.columns:  # Note: typo in original data
        features_df['payload_length'] = pd.to_numeric(
            df['lenght'], errors='coerce'
        ).fillna(0)
    elif 'length' in df.columns:
        features_df['payload_length'] = pd.to_numeric(
            df['length'], errors='coerce'
        ).fillna(0)
    else:
        features_df['payload_length'] = 0
    
    # Content length (from 'content' column if exists)
    if 'content' in df.columns:
        features_df['content_length'] = df['content'].astype(str).str.len()
    else:
        features_df['content_length'] = 0
    
    # URL length
    if 'URL' in df.columns:
        features_df['url_length'] = df['URL'].astype(str).str.len()
    else:
        features_df['url_length'] = 0
    
    # User-Agent length
    if 'User-Agent' in df.columns:
        features_df['user_agent_length'] = df['User-Agent'].astype(str).str.len()
    else:
        features_df['user_agent_length'] = 0
    
    # Cookie presence and length
    if 'cookie' in df.columns:
        features_df['has_cookie'] = (
            df['cookie'].notna() & (df['cookie'].astype(str).str.strip() != '')
        ).astype(int)
        features_df['cookie_length'] = df['cookie'].astype(str).str.len()
    
    # Accept-encoding compression flags
    if 'Accept-encoding' in df.columns:
        features_df['accepts_gzip'] = df['Accept-encoding'].astype(str).str.contains(
            'gzip', case=False, na=False
        ).astype(int)
        features_df['accepts_deflate'] = df['Accept-encoding'].astype(str).str.contains(
            'deflate', case=False, na=False
        ).astype(int)
    
    # Connection type
    if 'connection' in df.columns:
        features_df['connection_close'] = (
            df['connection'].astype(str).str.contains('close', case=False, na=False)
        ).astype(int)
        features_df['connection_keep_alive'] = (
            df['connection'].astype(str).str.contains('keep-alive', case=False, na=False)
        ).astype(int)
    
    # Security headers (commonly missing in attacks)
    security_headers = ['Content-Security-Policy', 'X-Frame-Options', 
                       'X-XSS-Protection', 'Strict-Transport-Security']
    for sec_header in security_headers:
        header_name = f'has_{sec_header.lower().replace("-", "_")}'
        if sec_header in df.columns:
            features_df[header_name] = (
                df[sec_header].notna() & (df[sec_header].astype(str).str.strip() != '')
            ).astype(int)
        else:
            features_df[header_name] = 0
    
    logger.info(f"Created {len(features_df.columns)} features")
    logger.info(f"Feature names: {features_df.columns.tolist()[:10]}...")
    
    return features_df


def train_isolation_forest(X_train):
    """
    Train Isolation Forest on normal traffic patterns.
    
    The model learns what 'normal' header configurations look like,
    and will flag deviations as anomalies.
    """
    logger.info("Training Isolation Forest model...")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Features: {X_train.shape[1]}")
    
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
    model.fit(X_train)
    
    logger.info("Model training completed")
    
    # Get predictions on training data for statistics
    train_predictions = model.predict(X_train)
    train_scores = model.score_samples(X_train)
    
    normal_count = (train_predictions == 1).sum()
    anomaly_count = (train_predictions == -1).sum()
    
    logger.info(f"Training set predictions:")
    logger.info(f"  Normal: {normal_count} ({normal_count/len(train_predictions)*100:.2f}%)")
    logger.info(f"  Anomalies: {anomaly_count} ({anomaly_count/len(train_predictions)*100:.2f}%)")
    logger.info(f"  Score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    
    return model


def save_model_and_artifacts(model, scaler, feature_columns):
    """Save trained model, scaler, and feature columns."""
    logger.info("\nSaving model and artifacts...")
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved as '{MODEL_FILE}'")
    
    # Save scaler
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved as '{SCALER_FILE}'")
    
    # Save feature columns
    with open(FEATURE_COLUMNS_FILE, 'wb') as f:
        pickle.dump(feature_columns, f)
    logger.info(f"Feature columns saved as '{FEATURE_COLUMNS_FILE}'")
    
    logger.info("\nAll artifacts saved successfully!")


def analyze_headers(headers_dict):
    """
    Analyze HTTP headers and return anomaly score.
    
    Args:
        headers_dict: Dictionary of HTTP headers
        
    Returns:
        dict: Contains anomaly score, prediction, and risk level
    """
    # Load model and artifacts
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_FILE, 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Create feature vector from headers
    features = {}
    
    # Header presence flags
    header_mapping = {
        'method': 'Method',
        'user-agent': 'User-Agent',
        'pragma': 'Pragma',
        'cache-control': 'Cache-Control',
        'accept': 'Accept',
        'accept-encoding': 'Accept-encoding',
        'accept-charset': 'Accept-charset',
        'language': 'language',
        'host': 'host',
        'cookie': 'cookie',
        'content-type': 'content-type',
        'connection': 'connection'
    }
    
    for key, header in header_mapping.items():
        features[f'has_{key.replace("-", "_")}'] = int(
            header.lower() in [k.lower() for k in headers_dict.keys()]
        )
    
    # Method type flags
    method = headers_dict.get('Method', headers_dict.get('method', '')).upper()
    for m in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']:
        features[f'method_{m.lower()}'] = int(method == m)
    
    # Numerical features
    features['header_count'] = len(headers_dict)
    features['payload_length'] = int(headers_dict.get('Content-Length', 
                                                       headers_dict.get('content-length', 0)))
    features['content_length'] = len(str(headers_dict.get('content', '')))
    features['url_length'] = len(str(headers_dict.get('URL', headers_dict.get('url', ''))))
    features['user_agent_length'] = len(str(headers_dict.get('User-Agent', 
                                                              headers_dict.get('user-agent', ''))))
    
    # Cookie features
    cookie = headers_dict.get('cookie', headers_dict.get('Cookie', ''))
    features['has_cookie'] = int(bool(cookie))
    features['cookie_length'] = len(str(cookie))
    
    # Encoding flags
    accept_encoding = str(headers_dict.get('Accept-encoding', 
                                           headers_dict.get('accept-encoding', ''))).lower()
    features['accepts_gzip'] = int('gzip' in accept_encoding)
    features['accepts_deflate'] = int('deflate' in accept_encoding)
    
    # Connection flags
    connection = str(headers_dict.get('connection', 
                                      headers_dict.get('Connection', ''))).lower()
    features['connection_close'] = int('close' in connection)
    features['connection_keep_alive'] = int('keep-alive' in connection)
    
    # Security headers
    security_headers = ['Content-Security-Policy', 'X-Frame-Options', 
                       'X-XSS-Protection', 'Strict-Transport-Security']
    for sec_header in security_headers:
        key = f'has_{sec_header.lower().replace("-", "_")}'
        features[key] = int(sec_header.lower() in [k.lower() for k in headers_dict.keys()])
    
    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([features])
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[feature_columns]
    
    # Scale features
    X = scaler.transform(feature_df)
    
    # Get prediction and score
    prediction = model.predict(X)[0]
    score = model.score_samples(X)[0]
    
    # Determine risk level
    if prediction == 1:
        risk_level = "Normal"
    else:
        if score < -0.5:
            risk_level = "High Risk"
        elif score < -0.2:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
    
    return {
        'anomaly_score': float(score),
        'is_anomaly': prediction == -1,
        'risk_level': risk_level,
        'prediction': 'Anomaly' if prediction == -1 else 'Normal'
    }


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("HTTP Header Anomaly Detection Model Training")
    logger.info("Testify by Trustify - Cybersecurity Project")
    logger.info("="*60)
    
    # Step 1: Load data
    df = load_data('data/http_headers.csv')
    
    # Step 2: Check classification column
    # The first column (Unnamed: 0) contains Normal/Anomaly labels
    first_col = df.columns[0]
    class_dist = df[first_col].value_counts()
    logger.info(f"\nClassification distribution ({first_col}):\n{class_dist}")
    
    # Step 3: Filter only Normal traffic for training
    logger.info("\nFiltering Normal traffic for training...")
    normal_df = df[df[first_col].astype(str).str.contains('Normal', case=False, na=False)]
    
    logger.info(f"Normal traffic samples: {len(normal_df)}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Using {len(normal_df)/len(df)*100:.2f}% of data for training")
    
    # Step 4: Engineer features
    features_df = engineer_features(normal_df)
    
    # Step 5: Scale features
    logger.info("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    # Step 6: Train Isolation Forest on Normal traffic only
    model = train_isolation_forest(X_scaled)
    
    # Step 7: Save model and artifacts
    save_model_and_artifacts(model, scaler, features_df.columns.tolist())
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved as: {MODEL_FILE}")
    logger.info("="*60)
    
    # Step 8: Demonstrate usage
    logger.info("\n" + "="*60)
    logger.info("Example Usage:")
    logger.info("="*60)
    
    # Example normal headers
    example_normal = {
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-encoding': 'gzip, deflate',
        'connection': 'keep-alive',
        'host': 'example.com',
        'URL': 'http://example.com/page.html'
    }
    
    result = analyze_headers(example_normal)
    logger.info(f"\nNormal Headers Analysis:")
    logger.info(f"  Anomaly Score: {result['anomaly_score']:.4f}")
    logger.info(f"  Prediction: {result['prediction']}")
    logger.info(f"  Risk Level: {result['risk_level']}")
    
    # Example suspicious headers
    example_suspicious = {
        'Method': 'GET',
        'User-Agent': 'sqlmap/1.0',
        'Accept': '*/*',
        'URL': 'http://example.com/admin.php?id=1 UNION SELECT'
    }
    
    result = analyze_headers(example_suspicious)
    logger.info(f"\nSuspicious Headers Analysis:")
    logger.info(f"  Anomaly Score: {result['anomaly_score']:.4f}")
    logger.info(f"  Prediction: {result['prediction']}")
    logger.info(f"  Risk Level: {result['risk_level']}")


if __name__ == "__main__":
    main()
