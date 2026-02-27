"""
HTTP Header Anomaly Detector - Standalone Usage Script
Testify by Trustify - Cybersecurity Project

Load the trained model and analyze HTTP headers for anomalies.
"""

import pickle
import pandas as pd
import numpy as np


class HeaderAnomalyDetector:
    """
    HTTP Header Anomaly Detector using Isolation Forest.
    Detects suspicious header patterns that deviate from normal traffic.
    """
    
    def __init__(self, model_path='header_analyzer_model.pkl',
                 scaler_path='header_scaler.pkl',
                 features_path='header_feature_columns.pkl'):
        """Load trained model and artifacts."""
        print("Loading HTTP Header Anomaly Detector...")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        print(f"Model loaded successfully with {len(self.feature_columns)} features")
    
    def extract_features(self, headers_dict):
        """Extract numerical features from HTTP headers dictionary."""
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
        
        # Normalize header keys to lowercase for matching
        headers_lower = {k.lower(): v for k, v in headers_dict.items()}
        
        for key, header in header_mapping.items():
            features[f'has_{key.replace("-", "_")}'] = int(
                header.lower() in headers_lower
            )
        
        # Method type flags
        method = headers_lower.get('method', '').upper()
        for m in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']:
            features[f'method_{m.lower()}'] = int(method == m)
        
        # Numerical features
        features['header_count'] = len(headers_dict)
        features['payload_length'] = int(headers_lower.get('content-length', 0))
        features['content_length'] = len(str(headers_dict.get('content', '')))
        features['url_length'] = len(str(headers_lower.get('url', '')))
        features['user_agent_length'] = len(str(headers_lower.get('user-agent', '')))
        
        # Cookie features
        cookie = headers_lower.get('cookie', '')
        features['has_cookie'] = int(bool(cookie))
        features['cookie_length'] = len(str(cookie))
        
        # Encoding flags
        accept_encoding = str(headers_lower.get('accept-encoding', '')).lower()
        features['accepts_gzip'] = int('gzip' in accept_encoding)
        features['accepts_deflate'] = int('deflate' in accept_encoding)
        
        # Connection flags
        connection = str(headers_lower.get('connection', '')).lower()
        features['connection_close'] = int('close' in connection)
        features['connection_keep_alive'] = int('keep-alive' in connection)
        
        # Security headers
        security_headers = ['content-security-policy', 'x-frame-options', 
                           'x-xss-protection', 'strict-transport-security']
        for sec_header in security_headers:
            key = f'has_{sec_header.replace("-", "_")}'
            features[key] = int(sec_header in headers_lower)
        
        return features
    
    def analyze(self, headers_dict):
        """
        Analyze HTTP headers and return anomaly score.
        
        Args:
            headers_dict: Dictionary of HTTP headers
            
        Returns:
            dict: Contains anomaly score, prediction, and risk level
        """
        # Extract features
        features = self.extract_features(headers_dict)
        
        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df[self.feature_columns]
        
        # Scale features
        X = self.scaler.transform(feature_df)
        
        # Get prediction and score
        prediction = self.model.predict(X)[0]
        score = self.model.score_samples(X)[0]
        
        # Determine risk level based on anomaly score
        if prediction == 1:
            risk_level = "Normal"
            threat_level = 0
        else:
            if score < -0.5:
                risk_level = "High Risk"
                threat_level = 3
            elif score < -0.2:
                risk_level = "Medium Risk"
                threat_level = 2
            else:
                risk_level = "Low Risk"
                threat_level = 1
        
        return {
            'anomaly_score': float(score),
            'is_anomaly': bool(prediction == -1),
            'risk_level': risk_level,
            'threat_level': threat_level,
            'prediction': 'Anomaly' if prediction == -1 else 'Normal'
        }
    
    def batch_analyze(self, headers_list):
        """
        Analyze multiple HTTP header sets.
        
        Args:
            headers_list: List of header dictionaries
            
        Returns:
            list: Analysis results for each header set
        """
        results = []
        for i, headers in enumerate(headers_list):
            result = self.analyze(headers)
            result['index'] = i
            results.append(result)
        return results


def demo():
    """Demonstration of the Header Anomaly Detector."""
    print("="*70)
    print("HTTP Header Anomaly Detector - Demo")
    print("Testify by Trustify - Cybersecurity Project")
    print("="*70)
    
    # Initialize detector
    detector = HeaderAnomalyDetector()
    
    # Test cases
    test_cases = [
        {
            'name': 'Normal Browser Request',
            'headers': {
                'Method': 'GET',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-encoding': 'gzip, deflate, br',
                'Accept-charset': 'utf-8',
                'connection': 'keep-alive',
                'host': 'www.example.com',
                'Cache-Control': 'no-cache',
                'URL': 'http://www.example.com/index.html'
            }
        },
        {
            'name': 'SQL Injection Attempt',
            'headers': {
                'Method': 'GET',
                'User-Agent': 'sqlmap/1.0',
                'Accept': '*/*',
                'URL': "http://site.com/page.php?id=1' OR '1'='1"
            }
        },
        {
            'name': 'XSS Attack Attempt',
            'headers': {
                'Method': 'GET',
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'text/html',
                'URL': 'http://site.com/search?q=<script>alert("XSS")</script>'
            }
        },
        {
            'name': 'Path Traversal Attempt',
            'headers': {
                'Method': 'GET',
                'User-Agent': 'curl/7.68.0',
                'Accept': '*/*',
                'URL': 'http://site.com/../../etc/passwd'
            }
        },
        {
            'name': 'Normal API Request',
            'headers': {
                'Method': 'POST',
                'User-Agent': 'python-requests/2.28.0',
                'Accept': 'application/json',
                'content-type': 'application/json',
                'Content-Length': '156',
                'connection': 'close',
                'host': 'api.example.com',
                'URL': 'https://api.example.com/v1/users'
            }
        }
    ]
    
    print("\n" + "="*70)
    print("Analyzing Test Cases:")
    print("="*70)
    
    for test in test_cases:
        print(f"\n📋 Test Case: {test['name']}")
        print("-" * 70)
        
        result = detector.analyze(test['headers'])
        
        # Display results
        print(f"  Prediction: {result['prediction']}")
        print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Threat Level: {result['threat_level']}/3")
        
        if result['is_anomaly']:
            print(f"  ⚠️  WARNING: Anomalous traffic detected!")
        else:
            print(f"  ✅ Traffic appears normal")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    demo()
