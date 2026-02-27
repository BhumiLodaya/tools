# HTTP Header Anomaly Detection

**Testify by Trustify - Cybersecurity Project**

An Isolation Forest-based machine learning system for detecting anomalous HTTP header patterns that may indicate cyberattacks, malicious bots, or suspicious traffic.

## Overview

This tool trains an Isolation Forest model on **normal HTTP traffic patterns** to learn what legitimate header configurations look like. Any deviation from these patterns is flagged as an anomaly, helping detect:

- SQL Injection attempts
- XSS (Cross-Site Scripting) attacks
- Path traversal attempts
- Malicious bots and scrapers
- Unusual request patterns
- Missing security headers

## Features

- **Automatic Feature Engineering**: Converts HTTP headers into 30+ numerical features
- **Anomaly Detection**: Uses Isolation Forest algorithm trained on normal traffic
- **Risk Scoring**: Provides anomaly scores and risk levels (Low/Medium/High)
- **Easy Integration**: Simple API for analyzing individual or batch requests
- **Reproducible**: Fixed random state for consistent results

## Installation

Install required dependencies:

```bash
pip install -r requirements-header-detector.txt
```

## Project Structure

```
├── data/
│   └── http_headers.csv          # HTTP header dataset
├── train_header_anomaly_detector.py  # Training script
├── header_detector.py             # Standalone usage script
├── header_analyzer_model.pkl     # Trained model (after training)
├── header_scaler.pkl             # Feature scaler (after training)
├── header_feature_columns.pkl    # Feature column names (after training)
└── README-header-detector.md     # This file
```

## Usage

### 1. Training the Model

Run the training script to train on normal HTTP traffic:

```bash
python train_header_anomaly_detector.py
```

This will:
1. Load HTTP headers from `data/http_headers.csv`
2. Filter **only normal traffic** for training
3. Engineer 30+ features from headers
4. Train Isolation Forest model
5. Save model, scaler, and feature columns

**Output files:**
- `header_analyzer_model.pkl` - Trained Isolation Forest model
- `header_scaler.pkl` - StandardScaler for features
- `header_feature_columns.pkl` - Feature column definitions

### 2. Using the Detector

#### Quick Demo

Run the demo script to see example analyses:

```bash
python header_detector.py
```

#### In Your Code

```python
from header_detector import HeaderAnomalyDetector

# Initialize detector
detector = HeaderAnomalyDetector()

# Analyze HTTP headers
headers = {
    'Method': 'GET',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'text/html',
    'Accept-encoding': 'gzip, deflate',
    'connection': 'keep-alive',
    'host': 'example.com',
    'URL': 'http://example.com/page.html'
}

result = detector.analyze(headers)

print(f"Prediction: {result['prediction']}")
print(f"Anomaly Score: {result['anomaly_score']:.4f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Is Anomaly: {result['is_anomaly']}")
```

#### Batch Analysis

```python
# Analyze multiple requests
headers_list = [headers1, headers2, headers3, ...]
results = detector.batch_analyze(headers_list)

for result in results:
    if result['is_anomaly']:
        print(f"⚠️  Anomaly detected in request {result['index']}")
        print(f"   Risk Level: {result['risk_level']}")
```

## How It Works

### Feature Engineering

The system extracts these feature types from HTTP headers:

1. **Binary Flags** (presence/absence):
   - Standard headers (User-Agent, Accept, Cookie, etc.)
   - Security headers (CSP, X-Frame-Options, etc.)
   - Method types (GET, POST, PUT, DELETE, etc.)

2. **Numerical Features**:
   - Header count
   - Payload length
   - Content length
   - URL length
   - User-Agent length
   - Cookie length

3. **Specific Patterns**:
   - Compression support (gzip, deflate)
   - Connection type (close, keep-alive)
   - Security header presence

### Anomaly Detection

**Isolation Forest** works by:
1. Learning patterns from normal traffic during training
2. Isolating anomalies by requiring fewer random partitions
3. Scoring each request (lower scores = more anomalous)

**Risk Levels:**
- **Normal**: Score indicates typical traffic pattern
- **Low Risk**: Minor deviation (score: -0.2 to 0)
- **Medium Risk**: Significant deviation (score: -0.5 to -0.2)
- **High Risk**: Severe deviation (score: < -0.5)

## Example Results

```
📋 Test Case: Normal Browser Request
  Prediction: Normal
  Anomaly Score: 0.0523
  Risk Level: Normal
  ✅ Traffic appears normal

📋 Test Case: SQL Injection Attempt
  Prediction: Anomaly
  Anomaly Score: -0.6234
  Risk Level: High Risk
  ⚠️  WARNING: Anomalous traffic detected!
```

## Data Format

The training data should be a CSV with HTTP header columns:

```csv
Method,User-Agent,Accept,Accept-encoding,host,cookie,connection,URL,classification
GET,Mozilla/5.0,...,gzip,example.com,session=abc,keep-alive,http://...,Normal
GET,sqlmap/1.0,...,*,target.com,,,http://...?id=1',Anomaly
```

Key columns:
- `classification`: Must contain "Normal" for training samples
- HTTP headers as separate columns
- `URL`: Request URL (optional but recommended)

## Configuration

Edit these parameters in `train_header_anomaly_detector.py`:

```python
RANDOM_STATE = 42           # For reproducibility
CONTAMINATION = 0.1         # Expected proportion of outliers (10%)
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request
from header_detector import HeaderAnomalyDetector

app = Flask(__name__)
detector = HeaderAnomalyDetector()

@app.before_request
def check_headers():
    headers = dict(request.headers)
    headers['Method'] = request.method
    headers['URL'] = request.url
    
    result = detector.analyze(headers)
    
    if result['threat_level'] >= 3:
        return "Suspicious activity detected", 403
```

### Log File Analysis

```python
import json
from header_detector import HeaderAnomalyDetector

detector = HeaderAnomalyDetector()

# Parse access logs
with open('access.log', 'r') as f:
    for line in f:
        log_entry = json.loads(line)
        result = detector.analyze(log_entry['headers'])
        
        if result['is_anomaly']:
            print(f"Anomaly: {log_entry['ip']} - {result['risk_level']}")
```

## Performance

- **Training Time**: ~30 seconds on 40K normal samples
- **Inference**: <1ms per request
- **Accuracy**: Depends on training data quality

## Limitations

- Requires representative normal traffic for training
- May flag legitimate but unusual traffic as anomalous
- Performance depends on feature engineering quality
- Should be used as part of a defense-in-depth strategy

## Contributing to Testify by Trustify

This is a standalone tool for the Testify by Trustify cybersecurity project. Suggestions for improvement:

- Add more security header checks
- Implement active learning for model updates
- Add support for HTTP/2 and HTTP/3 headers
- Create visualization dashboard for anomaly trends

## License

Part of the Testify by Trustify cybersecurity project.

## References

- Isolation Forest: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)
- HTTP Security Headers: OWASP Security Headers Project
- Anomaly Detection in Web Traffic: Various CSIC datasets
