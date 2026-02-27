# Hash Algorithm Identifier

**Part of Testify by Trustify - Cybersecurity Project**

A Machine Learning-based Hash Algorithm Identification system using Random Forest Classifier to automatically identify hash algorithms (MD5, SHA-1, SHA-256) from hash strings.

## 🎯 Overview

This tool analyzes hash strings and identifies which cryptographic hash algorithm was used to generate them. It uses feature engineering and Random Forest classification to achieve **100% accuracy** in distinguishing between MD5, SHA-1, and SHA-256 hashes.

## 🔑 Key Features

- **Random Forest Classifier** with 100% test accuracy
- **10 Engineered Features**: Length, character distribution, entropy analysis
- **predict_hash() Function**: Simple API for instant hash identification
- **Batch Processing**: Analyze multiple hashes efficiently
- **Validation**: Verify hash format and length correctness
- **30,000 Training Samples**: 10,000 samples per algorithm

## 📊 Supported Hash Algorithms

| Algorithm | Hash Length | Example |
|-----------|-------------|---------|
| **MD5** | 32 characters | `5d41402abc4b2a76b9719d911017c592` |
| **SHA-1** | 40 characters | `aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d` |
| **SHA-256** | 64 characters | `2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824` |

## 🚀 Quick Start

### 1. Train the Model (Optional - Pre-trained model included)

```bash
python train_hash_identifier.py
```

**Generated files:**
- `hash_id_model.pkl` - Trained Random Forest model
- `hash_id_features.pkl` - Feature column names
- `hash_id_metrics.pkl` - Performance metrics
- `hash_dataset.csv` - Raw hash dataset (30,000 samples)
- `hash_dataset_with_features.csv` - Dataset with extracted features

**Training Performance:**
- Training samples: 24,000 (80%)
- Test samples: 6,000 (20%)
- Training accuracy: **100.00%**
- Test accuracy: **100.00%**
- Training time: ~3 seconds

### 2. Use the predict_hash() Function

```python
from hash_identifier import predict_hash

# Identify a hash algorithm
predict_hash('5d41402abc4b2a76b9719d911017c592')

# Output:
# ======================================================================
# Hash Algorithm Identification
# ======================================================================
# Input Hash: 5d41402abc4b2a76b9719d911017c592
# Hash Length: 32 characters
#
# Predicted Algorithm: MD5
# Confidence Score: 100.00%
# Expected Length: 32 characters
#
# Probability Breakdown:
#   MD5     : 100.00% ████████████████████████████████████████████
#   SHA1    :   0.00%
#   SHA256  :   0.00%
#
# Validation:
#   Valid hex format: ✓
#   Length matches: ✓
#   Overall valid: ✓
# ======================================================================
```

### 3. Python API Usage

```python
from hash_identifier import HashIdentifier

# Initialize
identifier = HashIdentifier()

# Single prediction
result = identifier.predict('aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d')
print(f"Algorithm: {result['algorithm']}")
print(f"Confidence: {result['confidence']:.2f}%")

# Output:
# Algorithm: SHA1
# Confidence: 100.00%

# Detailed result
print(result)
# {
#     'algorithm': 'SHA1',
#     'confidence': 100.0,
#     'probabilities': {'MD5': 0.0, 'SHA1': 100.0, 'SHA256': 0.0},
#     'hash_length': 40,
#     'expected_length': 40
# }

# Batch prediction
hashes = [
    '5d41402abc4b2a76b9719d911017c592',  # MD5
    'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d',  # SHA1
    '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'  # SHA256
]

results = identifier.batch_predict(hashes)
for result in results:
    print(f"{result['hash']}: {result['algorithm']} ({result['confidence']:.1f}%)")

# Validate hash format
validation = identifier.validate_hash(
    '5d41402abc4b2a76b9719d911017c592',
    'MD5'
)
print(f"Valid: {validation['is_valid']}")
```

## 📋 Feature Engineering

The model extracts **10 features** from each hash string:

### Feature Details:

1. **hash_length** (int): Total length of hash string
   - MD5: 32, SHA-1: 40, SHA-256: 64
   - **Most important feature**: 47.32% importance

2. **num_digits** (int): Count of numeric digits (0-9)
   - Importance: 23.27%

3. **num_letters** (int): Count of alphabetic characters
   - Importance: 7.57%

4. **num_lowercase** (int): Count of lowercase letters
   - Importance: 6.97%

5. **num_uppercase** (int): Count of uppercase letters
   - Importance: 0.00% (hashes are typically lowercase)

6. **num_hex_letters** (int): Count of hex letters (a-f, A-F)
   - Importance: 8.74%

7. **entropy** (float): Shannon entropy of the string
   - Measures randomness/unpredictability
   - Importance: 0.94%

8. **digit_ratio** (float): Ratio of digits to total length
   - Importance: 1.68%

9. **letter_ratio** (float): Ratio of letters to total length
   - Importance: 2.05%

10. **hex_letter_ratio** (float): Ratio of hex letters to total
    - Importance: 1.47%

### Feature Importance Ranking:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | hash_length | 47.32% |
| 2 | num_digits | 23.27% |
| 3 | num_hex_letters | 8.74% |
| 4 | num_letters | 7.57% |
| 5 | num_lowercase | 6.97% |
| 6 | letter_ratio | 2.05% |
| 7 | digit_ratio | 1.68% |
| 8 | hex_letter_ratio | 1.47% |
| 9 | entropy | 0.94% |
| 10 | num_uppercase | 0.00% |

## 🧪 Model Performance

### Classification Report:

```
              precision    recall  f1-score   support

         MD5       1.00      1.00      1.00      2000
        SHA1       1.00      1.00      1.00      2000
      SHA256       1.00      1.00      1.00      2000

    accuracy                           1.00      6000
   macro avg       1.00      1.00      1.00      6000
weighted avg       1.00      1.00      1.00      6000
```

### Confusion Matrix:

```
             Predicted
             MD5   SHA1  SHA256
Actual MD5   2000    0      0
      SHA1      0 2000      0
    SHA256      0    0   2000
```

**Perfect classification** - Zero misclassifications!

### Why 100% Accuracy?

Hash algorithms produce deterministic, fixed-length outputs:
- MD5 **always** produces 32-character hexadecimal strings
- SHA-1 **always** produces 40-character hexadecimal strings
- SHA-256 **always** produces 64-character hexadecimal strings

The `hash_length` feature alone is nearly sufficient for perfect classification. Additional features provide robustness against edge cases and validate the hash format.

## 📡 Integration Examples

### Flask REST API

```python
from flask import Flask, request, jsonify
from hash_identifier import HashIdentifier

app = Flask(__name__)
identifier = HashIdentifier()

@app.route('/api/identify-hash', methods=['POST'])
def identify_hash():
    """Identify hash algorithm from hash string."""
    try:
        hash_string = request.json.get('hash')
        
        if not hash_string:
            return jsonify({'error': 'No hash provided'}), 400
        
        result = identifier.predict(hash_string)
        validation = identifier.validate_hash(hash_string, result['algorithm'])
        
        return jsonify({
            'algorithm': result['algorithm'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'hash_length': result['hash_length'],
            'expected_length': result['expected_length'],
            'is_valid': validation['is_valid']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Command Line Usage

```bash
# Using Python directly
python -c "from hash_identifier import predict_hash; predict_hash('YOUR_HASH_HERE')"

# Test script
python test_hash_identifier.py

# Full demo
python hash_identifier.py
```

### React Frontend Integration

```jsx
import React, { useState } from 'react';

function HashIdentifier() {
  const [hash, setHash] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const identifyHash = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/identify-hash', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hash })
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Hash identification failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="hash-identifier">
      <h2>Hash Algorithm Identifier</h2>
      
      <input
        type="text"
        placeholder="Enter hash string..."
        value={hash}
        onChange={(e) => setHash(e.target.value)}
      />
      
      <button onClick={identifyHash} disabled={loading}>
        {loading ? 'Analyzing...' : 'Identify Hash'}
      </button>
      
      {result && (
        <div className="result">
          <h3>Algorithm: {result.algorithm}</h3>
          <p>Confidence: {result.confidence.toFixed(2)}%</p>
          <p>Hash Length: {result.hash_length} chars</p>
          <p>Expected: {result.expected_length} chars</p>
          <p>Valid: {result.is_valid ? '✓' : '✗'}</p>
        </div>
      )}
    </div>
  );
}

export default HashIdentifier;
```

## 🛠️ Technology Stack

- **Machine Learning**: scikit-learn Random Forest Classifier
- **Data Processing**: pandas, numpy
- **Hashing**: hashlib (MD5, SHA-1, SHA-256)
- **Model Persistence**: joblib
- **Python Version**: 3.8+

## 📦 Dependencies

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
joblib>=1.2.0
```

Install with:
```bash
pip install -r requirements-hash-identifier.txt
```

## 🔬 Use Cases

### 1. Security Analysis
Identify hash algorithms used in password databases, configuration files, or log files.

### 2. Forensics
Determine hash types in digital forensics investigations.

### 3. Password Cracking
Identify hash algorithm before attempting to crack hashed passwords.

### 4. Code Review
Automatically detect hash algorithms in source code or documentation.

### 5. Security Audits
Identify weak hash algorithms (MD5, SHA-1) that should be upgraded to SHA-256 or better.

### 6. Educational Tools
Teach cryptography and hash function concepts with interactive demonstrations.

## 📈 Example Test Cases

### Test Case 1: MD5 Hash (32 chars)
```python
predict_hash('5d41402abc4b2a76b9719d911017c592')
# Result: MD5 (100.00% confidence)
```

### Test Case 2: SHA-1 Hash (40 chars)
```python
predict_hash('aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d')
# Result: SHA1 (100.00% confidence)
```

### Test Case 3: SHA-256 Hash (64 chars)
```python
predict_hash('2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')
# Result: SHA256 (100.00% confidence)
```

### Test Case 4: Batch Processing
```python
from hash_identifier import HashIdentifier
import hashlib

identifier = HashIdentifier()

# Generate test hashes
test_data = ["password", "admin", "trustify"]
hashes = []

for text in test_data:
    hashes.append(hashlib.md5(text.encode()).hexdigest())
    hashes.append(hashlib.sha1(text.encode()).hexdigest())
    hashes.append(hashlib.sha256(text.encode()).hexdigest())

# Batch predict
results = identifier.batch_predict(hashes)

for result in results:
    print(f"{result['algorithm']}: {result['confidence']:.1f}%")

# Output:
# MD5: 100.0%
# SHA1: 100.0%
# SHA256: 100.0%
# ... (9 total results)
```

## 🎓 How It Works

### 1. Data Generation
```python
# Generate 10,000 random strings
# Hash each with MD5, SHA-1, SHA-256
# Total: 30,000 hash samples
```

### 2. Feature Extraction
```python
# Extract features from each hash:
# - Length (most discriminative)
# - Character distribution
# - Entropy
# - Ratios
```

### 3. Model Training
```python
# Train Random Forest Classifier
# 100 trees, max_depth=20
# 80/20 train/test split
# Perfect accuracy on test set
```

### 4. Prediction
```python
# Input hash → Extract features → Predict algorithm
# Return confidence scores for each algorithm
```

## 🌟 Advantages

✅ **Perfect Accuracy**: 100% test accuracy on all hash types  
✅ **Fast**: Predictions in <10ms  
✅ **Simple API**: Just call `predict_hash(hash_string)`  
✅ **Robust**: Validates hash format and length  
✅ **Extensible**: Easy to add more hash algorithms (SHA-512, etc.)  
✅ **Educational**: Clear feature engineering and model explanation  
✅ **Production-Ready**: Error handling, logging, validation  

## 🔮 Future Enhancements

- [ ] Support for SHA-512, BLAKE2, etc.
- [ ] Confidence threshold tuning
- [ ] Hash strength scoring
- [ ] Rainbow table detection
- [ ] Salted vs unsalted hash detection
- [ ] Web-based GUI interface
- [ ] API rate limiting

## 📝 Notes

### Why This Works:

Hash algorithms produce **deterministic, fixed-length** outputs:
- The length feature alone provides near-perfect discrimination
- Additional features add robustness and validation
- Random Forest handles the classification with ease

### Limitations:

- Only identifies MD5, SHA-1, and SHA-256 (can be extended)
- Assumes standard hexadecimal hash representation
- Won't work with base64-encoded or salted hashes
- Requires valid hash format (hexadecimal characters only)

### Security Note:

This tool **identifies** hash algorithms but does **not** crack or reverse hashes. It's designed for:
- Security analysis and auditing
- Educational purposes
- Hash format validation
- Forensics and research

## 📚 References

- **MD5**: RFC 1321 (128-bit, 32 hex chars)
- **SHA-1**: RFC 3174 (160-bit, 40 hex chars)
- **SHA-256**: FIPS 180-4 (256-bit, 64 hex chars)
- **Random Forest**: Breiman, 2001

## 👥 Authors

Built for the Testify by Trustify Cybersecurity Project

## 📧 Support

Check the demo script (`hash_identifier.py`) for complete usage examples and test cases.

---

**Built with ❤️ for the cybersecurity community**

Last Updated: February 2026  
Version: 1.0.0
