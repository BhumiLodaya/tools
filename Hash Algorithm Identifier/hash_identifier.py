"""
Hash Algorithm Identifier - Standalone Usage Script
Testify by Trustify - Cybersecurity Project

Load the trained model and predict hash algorithms from hash strings.
Includes the predict_hash() function as requested.
"""

import joblib
import pandas as pd
import numpy as np
import hashlib


class HashIdentifier:
    """
    Hash Algorithm Identifier using Random Forest Classifier.
    Identifies MD5, SHA-1, and SHA-256 hash algorithms from hash strings.
    """
    
    def __init__(self, model_path='hash_id_model.pkl', 
                 features_path='hash_id_features.pkl'):
        """Load trained model and feature columns."""
        print("Loading Hash Algorithm Identifier...")
        
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(features_path)
        
        print(f"Model loaded successfully")
        print(f"Supported algorithms: MD5, SHA1, SHA256")
        print(f"Features used: {len(self.feature_columns)}")
    
    def calculate_entropy(self, text):
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
    
    def extract_features(self, hash_string):
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
        entropy = self.calculate_entropy(hash_string)
        
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
    
    def predict(self, hash_string):
        """
        Predict the hash algorithm for a given hash string.
        
        Args:
            hash_string: The hash string to identify
            
        Returns:
            dict: Prediction results with algorithm, confidence, and probabilities
        """
        # Clean the hash string
        hash_string = hash_string.strip().lower()
        
        # Extract features
        features = self.extract_features(hash_string)
        features_df = pd.DataFrame([features])
        features_df = features_df[self.feature_columns]
        
        # Predict
        prediction = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        confidence = max(probabilities) * 100
        
        # Get probabilities for each class
        class_probabilities = {}
        for idx, class_name in enumerate(self.model.classes_):
            class_probabilities[class_name] = probabilities[idx] * 100
        
        return {
            'algorithm': prediction,
            'confidence': confidence,
            'probabilities': class_probabilities,
            'hash_length': len(hash_string),
            'expected_length': self._get_expected_length(prediction)
        }
    
    def _get_expected_length(self, algorithm):
        """Get expected hash length for an algorithm."""
        lengths = {
            'MD5': 32,
            'SHA1': 40,
            'SHA256': 64
        }
        return lengths.get(algorithm, 'Unknown')
    
    def validate_hash(self, hash_string, algorithm):
        """
        Validate if a hash string matches the expected format for an algorithm.
        
        Args:
            hash_string: The hash string to validate
            algorithm: Expected algorithm (MD5, SHA1, SHA256)
            
        Returns:
            dict: Validation results
        """
        expected_lengths = {
            'MD5': 32,
            'SHA1': 40,
            'SHA256': 64
        }
        
        hash_string = hash_string.strip().lower()
        actual_length = len(hash_string)
        expected_length = expected_lengths.get(algorithm.upper(), None)
        
        # Check if all characters are hexadecimal
        is_hex = all(c in '0123456789abcdef' for c in hash_string)
        
        # Check length match
        length_match = actual_length == expected_length if expected_length else False
        
        return {
            'is_valid': is_hex and length_match,
            'is_hex': is_hex,
            'length_match': length_match,
            'actual_length': actual_length,
            'expected_length': expected_length
        }
    
    def batch_predict(self, hash_list):
        """
        Predict algorithms for multiple hash strings.
        
        Args:
            hash_list: List of hash strings
            
        Returns:
            list: Prediction results for each hash
        """
        results = []
        for hash_string in hash_list:
            result = self.predict(hash_string)
            result['hash'] = hash_string[:20] + '...' if len(hash_string) > 20 else hash_string
            results.append(result)
        return results


def predict_hash(input_string):
    """
    Main prediction function as requested.
    Takes a raw hash and prints the predicted algorithm and confidence score.
    
    Args:
        input_string: Hash string to identify
    """
    # Load model
    identifier = HashIdentifier()
    
    # Predict
    result = identifier.predict(input_string)
    
    # Print results
    print("\n" + "="*70)
    print("Hash Algorithm Identification")
    print("="*70)
    print(f"Input Hash: {input_string}")
    print(f"Hash Length: {result['hash_length']} characters")
    print(f"\nPredicted Algorithm: {result['algorithm']}")
    print(f"Confidence Score: {result['confidence']:.2f}%")
    print(f"Expected Length: {result['expected_length']} characters")
    
    print(f"\nProbability Breakdown:")
    for algo, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob / 2)
        bar = '█' * bar_length
        print(f"  {algo:8s}: {prob:6.2f}% {bar}")
    
    # Validation
    validation = identifier.validate_hash(input_string, result['algorithm'])
    print(f"\nValidation:")
    print(f"  Valid hex format: {'✓' if validation['is_hex'] else '✗'}")
    print(f"  Length matches: {'✓' if validation['length_match'] else '✗'}")
    print(f"  Overall valid: {'✓' if validation['is_valid'] else '✗'}")
    
    print("="*70)
    
    return result


def demo():
    """Demonstration of the Hash Algorithm Identifier."""
    print("="*70)
    print("Hash Algorithm Identifier - Demo")
    print("Testify by Trustify - Cybersecurity Project")
    print("="*70)
    
    # Initialize identifier
    identifier = HashIdentifier()
    
    print("\n" + "="*70)
    print("Test Case 1: Known Hash Strings")
    print("="*70)
    
    # Test with known hashes of the word "hello"
    test_cases = [
        {
            'hash': '5d41402abc4b2a76b9719d911017c592',
            'expected': 'MD5',
            'description': 'MD5 hash of "hello"'
        },
        {
            'hash': 'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d',
            'expected': 'SHA1',
            'description': 'SHA-1 hash of "hello"'
        },
        {
            'hash': '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824',
            'expected': 'SHA256',
            'description': 'SHA-256 hash of "hello"'
        }
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\nTest {idx}: {test['description']}")
        print(f"Hash: {test['hash']}")
        print(f"Expected: {test['expected']}")
        
        result = identifier.predict(test['hash'])
        
        print(f"Predicted: {result['algorithm']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        
        match = result['algorithm'] == test['expected']
        print(f"Match: {'✓ CORRECT' if match else '✗ INCORRECT'}")
        
        if match:
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nTest Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    
    # Test Case 2: Fresh hashes
    print("\n" + "="*70)
    print("Test Case 2: Fresh Hash Generation")
    print("="*70)
    
    test_strings = ["password123", "SecureP@ssw0rd!", "admin"]
    
    for test_string in test_strings:
        print(f"\n--- Testing with: '{test_string}' ---")
        
        # Generate hashes
        md5_hash = hashlib.md5(test_string.encode()).hexdigest()
        sha1_hash = hashlib.sha1(test_string.encode()).hexdigest()
        sha256_hash = hashlib.sha256(test_string.encode()).hexdigest()
        
        hashes = [
            ('MD5', md5_hash),
            ('SHA1', sha1_hash),
            ('SHA256', sha256_hash)
        ]
        
        for expected, hash_val in hashes:
            result = identifier.predict(hash_val)
            match = '✓' if result['algorithm'] == expected else '✗'
            print(f"{expected:8s}: {result['algorithm']:8s} ({result['confidence']:5.1f}%) {match}")
    
    # Test Case 3: Batch prediction
    print("\n" + "="*70)
    print("Test Case 3: Batch Prediction")
    print("="*70)
    
    batch_hashes = [
        hashlib.md5(b"test1").hexdigest(),
        hashlib.sha1(b"test2").hexdigest(),
        hashlib.sha256(b"test3").hexdigest(),
        hashlib.md5(b"cybersecurity").hexdigest(),
        hashlib.sha256(b"Trustify").hexdigest(),
    ]
    
    print(f"\nAnalyzing {len(batch_hashes)} hashes...")
    results = identifier.batch_predict(batch_hashes)
    
    print(f"\n{'Hash':<25} {'Algorithm':<10} {'Confidence':<12} {'Length':<8}")
    print("-" * 70)
    for result in results:
        print(f"{result['hash']:<25} {result['algorithm']:<10} {result['confidence']:>5.1f}% {result['hash_length']:>12}")
    
    # Test Case 4: Using the predict_hash() function
    print("\n" + "="*70)
    print("Test Case 4: Using predict_hash() Function")
    print("="*70)
    
    # Example hash
    example_hash = "e99a18c428cb38d5f260853678922e03"  # MD5 of "abc123"
    predict_hash(example_hash)
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)
    
    print("\n" + "="*70)
    print("Usage Examples:")
    print("="*70)
    print("""
# Python API:
from hash_identifier import HashIdentifier

# Initialize
identifier = HashIdentifier()

# Single prediction
result = identifier.predict("5d41402abc4b2a76b9719d911017c592")
print(f"Algorithm: {result['algorithm']}")
print(f"Confidence: {result['confidence']:.2f}%")

# Using predict_hash() function
from hash_identifier import predict_hash
predict_hash("aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d")

# Batch prediction
hashes = ["hash1", "hash2", "hash3"]
results = identifier.batch_predict(hashes)
    """)


if __name__ == "__main__":
    demo()
