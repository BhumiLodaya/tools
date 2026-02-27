# SSL Security Scorer

**Testify by Trustify - Cybersecurity Project**

A Random Forest Regressor-based machine learning system for calculating SSL/TLS security scores (0-100) based on certificate configuration and best practices.

## Overview

This tool evaluates SSL/TLS certificates and assigns security scores based on multiple factors including expiration status, encryption strength, certificate authority validation, and revocation status. It helps identify weak or problematic SSL configurations before they become security vulnerabilities.

## Features

- **Automated Scoring**: Calculate security scores (0-100) for SSL certificates
- **Missing Value Handling**: Uses SimpleImputer for robust preprocessing
- **Feature Normalization**: StandardScaler for numerical features
- **Random Forest Model**: 100 decision trees for accurate predictions
- **Grading System**: A-F letter grades based on scores
- **Recommendations**: Actionable security advice for each certificate
- **Batch Processing**: Analyze multiple certificates simultaneously

## Scoring Factors

The model considers these key security factors:

1. **Days to Expiry** (25 points)
   - Expired certificates: 0 points
   - Valid certificates: Scaled by remaining time
   
2. **Self-Signed Status** (-30 points penalty)
   - CA-signed: Full points
   - Self-signed: Major penalty
   
3. **Cipher Bits** (20 points)
   - 56-bit or less: 0 points
   - 128-bit: 10 points
   - 192-bit: 15 points
   - 256-bit: 20 points
   
4. **Revocation Status** (-50 points penalty)
   - Not revoked: Full points
   - Revoked: Critical penalty

## Installation

Install required dependencies:

```bash
pip install -r requirements-ssl-scorer.txt
```

## Project Structure

```
├── data/
│   └── ssl_data.csv              # SSL certificate dataset
├── train_ssl_scorer.py           # Training script
├── ssl_scorer.py                 # Standalone usage script
├── ssl_scoring_model.pkl         # Trained model (after training)
├── ssl_imputer.pkl               # SimpleImputer (after training)
├── ssl_scaler.pkl                # StandardScaler (after training)
├── ssl_feature_names.pkl         # Feature definitions (after training)
└── README-ssl-scorer.md          # This file
```

## Usage

### 1. Training the Model

Run the training script:

```bash
python train_ssl_scorer.py
```

**What it does:**
1. Loads SSL certificate data from `data/ssl_data.csv`
2. Generates sample data if file doesn't exist
3. Creates security scores based on weighted best practices
4. Handles missing values with SimpleImputer
5. Normalizes features with StandardScaler
6. Trains Random Forest Regressor (100 estimators)
7. Saves model, imputer, scaler, and feature names

**Output files:**
- `ssl_scoring_model.pkl` - Trained Random Forest model
- `ssl_imputer.pkl` - SimpleImputer for missing values
- `ssl_scaler.pkl` - StandardScaler for normalization
- `ssl_feature_names.pkl` - Feature column definitions

### 2. Using the Scorer

#### Quick Demo

```bash
python ssl_scorer.py
```

#### In Your Code

```python
from ssl_scorer import SSLSecurityScorer

# Initialize scorer
scorer = SSLSecurityScorer()

# Score a single certificate
cert_features = {
    'days_to_expiry': 180,
    'is_self_signed': 0,
    'cipher_bits': 256,
    'is_revoked': 0
}

score = scorer.predict_score(cert_features)
print(f"Security Score: {score:.1f}/100")
```

#### Detailed Assessment

```python
# Get detailed assessment with recommendations
assessment = scorer.assess_certificate(cert_features)

print(f"Score: {assessment['score']:.1f}/100")
print(f"Grade: {assessment['grade']}")
print(f"Status: {assessment['status']}")
print("Recommendations:")
for rec in assessment['recommendations']:
    print(f"  {rec}")
```

#### Batch Scoring

```python
# Score multiple certificates
certificates = [
    {'days_to_expiry': 180, 'is_self_signed': 0, 'cipher_bits': 256, 'is_revoked': 0},
    {'days_to_expiry': 20, 'is_self_signed': 0, 'cipher_bits': 128, 'is_revoked': 0},
    {'days_to_expiry': -10, 'is_self_signed': 1, 'cipher_bits': 56, 'is_revoked': 0}
]

scores = scorer.batch_score(certificates)
for i, score in enumerate(scores):
    print(f"Certificate {i+1}: {score:.1f}/100")
```

## How It Works

### Feature Engineering

The model uses 4 core features:

1. **days_to_expiry** (int)
   - Days until certificate expires
   - Negative values indicate expired certificates
   
2. **is_self_signed** (binary: 0 or 1)
   - 0 = CA-signed (trusted)
   - 1 = Self-signed (not trusted)
   
3. **cipher_bits** (int: 56, 128, 192, 256)
   - Encryption strength in bits
   - Higher is better
   
4. **is_revoked** (binary: 0 or 1)
   - 0 = Valid certificate
   - 1 = Revoked (unsafe)

### Scoring Logic

**Base Score: 100 points**

Adjustments:
- ✅ Valid expiry (365+ days): +25 points
- ⚠️  Expiring soon (30-90 days): +10-20 points
- 🔴 Expired: 0 points
- ✅ CA-signed: Full points
- 🔴 Self-signed: -30 points
- ✅ 256-bit encryption: +20 points
- ⚠️  128-bit encryption: +10 points
- 🔴 <128-bit encryption: 0 points
- ✅ Not revoked: Full points
- 🔴 Revoked: -50 points

**Final Score: Clipped to [0, 100]**

### Grading System

- **A (90-100)**: Excellent - Strong security configuration
- **B (80-89)**: Good - Minor improvements recommended
- **C (70-79)**: Fair - Some security concerns
- **D (60-69)**: Poor - Upgrade needed
- **F (<60)**: Critical - Immediate action required

## Example Results

```
🔐 Test Case: Strong SSL Configuration (TLS 1.3, 256-bit)
  Security Score: 95.2/100
  Grade: A
  Status: Excellent
  Recommendations:
    ✅ Certificate configuration looks good

🔐 Test Case: Expired Certificate
  Security Score: 45.3/100
  Grade: F
  Status: Critical
  Recommendations:
    🔴 CRITICAL: Certificate has expired!

🔐 Test Case: Self-Signed Certificate
  Security Score: 68.7/100
  Grade: D
  Status: Poor
  Recommendations:
    🔴 Self-signed certificate - not trusted by browsers
```

## Configuration

Edit these parameters in `train_ssl_scorer.py`:

```python
N_ESTIMATORS = 100       # Number of trees in Random Forest
TEST_SIZE = 0.2          # 20% test split
RANDOM_STATE = 42        # For reproducibility
FEATURE_COLUMNS = [      # Features to use
    'days_to_expiry',
    'is_self_signed',
    'cipher_bits',
    'is_revoked'
]
```

## Integration Examples

### Web Application Integration

```python
from ssl_scorer import SSLSecurityScorer
import ssl
import socket

scorer = SSLSecurityScorer()

def check_website_ssl(hostname):
    # Get SSL certificate
    context = ssl.create_default_context()
    with socket.create_connection((hostname, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert = ssock.getpeercert()
    
    # Extract features
    features = {
        'days_to_expiry': calculate_days_to_expiry(cert),
        'is_self_signed': check_if_self_signed(cert),
        'cipher_bits': get_cipher_bits(ssock),
        'is_revoked': check_revocation(cert)
    }
    
    # Get score
    assessment = scorer.assess_certificate(features)
    return assessment
```

### Certificate Monitoring

```python
from ssl_scorer import SSLSecurityScorer
import schedule

scorer = SSLSecurityScorer()

def monitor_certificates():
    certificates = load_certificate_inventory()
    
    for cert in certificates:
        assessment = scorer.assess_certificate(cert)
        
        if assessment['grade'] in ['D', 'F']:
            send_alert(cert['hostname'], assessment)
        
        log_score(cert['hostname'], assessment['score'])

# Run daily
schedule.every().day.at("09:00").do(monitor_certificates)
```

### Security Dashboard

```python
from ssl_scorer import SSLSecurityScorer
import pandas as pd

scorer = SSLSecurityScorer()

def generate_ssl_report(certificate_df):
    # Score all certificates
    scores = scorer.batch_score(certificate_df)
    
    # Analyze fleet
    results = [{'score': s} for s in scores]
    summary = scorer.analyze_fleet(results)
    
    # Generate report
    report = {
        'total': summary['total_certificates'],
        'avg_score': summary['average_score'],
        'grade_distribution': summary['grade_distribution'],
        'critical_count': summary['failing']
    }
    
    return report
```

## Data Format

The training data should be a CSV with these columns:

```csv
days_to_expiry,is_self_signed,cipher_bits,is_revoked,security_score
180,0,256,0,95.2
-10,0,256,0,45.3
365,1,128,0,68.7
90,0,256,1,42.1
```

- `days_to_expiry`: Integer (negative if expired)
- `is_self_signed`: Binary (0 or 1)
- `cipher_bits`: Integer (56, 128, 192, or 256)
- `is_revoked`: Binary (0 or 1)
- `security_score`: Float (0-100, optional - calculated if missing)

## Performance

- **Training Time**: <5 seconds on 5,000 certificates
- **Inference**: <1ms per certificate
- **Model Size**: ~2MB
- **Accuracy**: RMSE <5 points, R² >0.95

## Best Practices

### High-Security SSL Configuration

✅ **Recommendations:**
- Use CA-signed certificates (not self-signed)
- Minimum 256-bit encryption
- Keep certificates valid (renew 30+ days before expiry)
- Monitor for revocation status
- Use TLS 1.2 or higher
- Enable Extended Validation (EV) when possible
- Use 2048-bit or 4096-bit RSA keys

### Common Issues Detected

🔴 **Critical:**
- Expired certificates
- Revoked certificates
- Self-signed certificates in production

⚠️ **Warnings:**
- Expiring within 30 days
- Weak encryption (<256-bit)
- Old TLS versions

## Limitations

- Requires accurate feature extraction from certificates
- Scoring weights are based on industry best practices
- Should be combined with other security tools
- Does not validate certificate chain

## Future Enhancements

- Add certificate chain validation
- Include OCSP stapling status
- Support for multiple cipher suites
- Integration with Certificate Transparency logs
- Automated remediation suggestions

## Contributing to Testify by Trustify

This is a standalone tool for the Testify by Trustify cybersecurity project.

## License

Part of the Testify by Trustify cybersecurity project.

## References

- Random Forest Regressor: Scikit-learn Documentation
- SSL/TLS Best Practices: OWASP, Mozilla SSL Config
- Certificate Authority Guidelines: CA/Browser Forum
