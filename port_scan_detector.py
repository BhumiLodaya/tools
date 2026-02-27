"""
Port Scan Anomaly Detector - Standalone Usage Script
Testify by Trustify - Cybersecurity Project

Load the trained model and detect port scanning activities in network traffic.
"""

import joblib
import pandas as pd
import numpy as np


class PortScanDetector:
    """
    Port Scan Anomaly Detector using Isolation Forest.
    Detects port scanning and other anomalous network patterns.
    """
    
    def __init__(self, model_path='port_scan_model.pkl',
                 scaler_path='port_scaler.pkl',
                 features_path='port_feature_names.pkl'):
        """Load trained model and artifacts."""
        print("Loading Port Scan Anomaly Detector...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        print(f"Model loaded successfully with {len(self.feature_names)} features")
    
    def detect(self, traffic_data):
        """
        Detect port scanning activity in network traffic.
        
        Args:
            traffic_data: Dictionary or DataFrame with network traffic features
            
        Returns:
            dict: Detection result with prediction and anomaly score
        """
        # Convert to DataFrame if dict
        if isinstance(traffic_data, dict):
            df = pd.DataFrame([traffic_data])
        else:
            df = traffic_data.copy()
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Handle inf and NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        score = self.model.score_samples(X_scaled)[0]
        
        # Determine severity
        if prediction == 1:
            severity = "Normal"
            risk = "Low"
            threat_level = 0
        else:
            if score < -0.3:
                severity = "Critical"
                risk = "High"
                threat_level = 3
            elif score < -0.15:
                severity = "Warning"
                risk = "Medium"
                threat_level = 2
            else:
                severity = "Alert"
                risk = "Low"
                threat_level = 1
        
        return {
            'prediction': int(prediction),
            'is_port_scan': bool(prediction == -1),
            'anomaly_score': float(score),
            'severity': severity,
            'risk_level': risk,
            'threat_level': threat_level,
            'status': 'Port Scan Detected' if prediction == -1 else 'Normal Traffic'
        }
    
    def batch_detect(self, traffic_list):
        """
        Detect port scanning in multiple traffic flows.
        
        Args:
            traffic_list: List of traffic data dictionaries or DataFrame
            
        Returns:
            list: Detection results for each flow
        """
        if isinstance(traffic_list, pd.DataFrame):
            df = traffic_list
        else:
            df = pd.DataFrame(traffic_list)
        
        results = []
        for idx in range(len(df)):
            result = self.detect(df.iloc[idx:idx+1])
            result['index'] = idx
            results.append(result)
        
        return results
    
    def analyze_traffic_summary(self, results):
        """
        Analyze batch detection results and provide summary.
        
        Args:
            results: List of detection results
            
        Returns:
            dict: Summary statistics
        """
        total = len(results)
        port_scans = sum(1 for r in results if r['is_port_scan'])
        critical = sum(1 for r in results if r['severity'] == 'Critical')
        warnings = sum(1 for r in results if r['severity'] == 'Warning')
        
        return {
            'total_flows': total,
            'normal': total - port_scans,
            'anomalies': port_scans,
            'critical': critical,
            'warnings': warnings,
            'anomaly_rate': port_scans / total if total > 0 else 0
        }


def demo():
    """Demonstration of the Port Scan Detector."""
    print("="*70)
    print("Port Scan Anomaly Detector - Demo")
    print("Testify by Trustify - Cybersecurity Project")
    print("="*70)
    
    # Initialize detector
    detector = PortScanDetector()
    
    # Test cases
    test_cases = [
        {
            'name': 'Normal HTTP Traffic',
            'data': {
                'Destination Port': 80,
                'Flow Duration': 25000,
                'Total Fwd Packets': 15,
                'Total Length of Fwd Packets': 8500,
                'Flow Bytes/s': 340,
                'Flow Packets/s': 0.6,
                'Fwd Packets/s': 0.6,
                'Bwd Packets/s': 0.4,
                'Packet Length Mean': 567,
                'Average Packet Size': 567,
                'FIN Flag Count': 1,
                'PSH Flag Count': 3,
                'ACK Flag Count': 12
            }
        },
        {
            'name': 'Normal HTTPS Traffic',
            'data': {
                'Destination Port': 443,
                'Flow Duration': 50000,
                'Total Fwd Packets': 25,
                'Total Length of Fwd Packets': 15000,
                'Flow Bytes/s': 300,
                'Flow Packets/s': 0.5,
                'Fwd Packets/s': 0.5,
                'Bwd Packets/s': 0.5,
                'Packet Length Mean': 600,
                'Average Packet Size': 600,
                'FIN Flag Count': 1,
                'PSH Flag Count': 5,
                'ACK Flag Count': 20
            }
        },
        {
            'name': 'SYN Port Scan (SSH)',
            'data': {
                'Destination Port': 22,
                'Flow Duration': 50,
                'Total Fwd Packets': 1,
                'Total Length of Fwd Packets': 60,
                'Flow Bytes/s': 1200,
                'Flow Packets/s': 20,
                'Fwd Packets/s': 20,
                'Bwd Packets/s': 0,
                'Packet Length Mean': 60,
                'Average Packet Size': 60,
                'FIN Flag Count': 0,
                'PSH Flag Count': 0,
                'ACK Flag Count': 0
            }
        },
        {
            'name': 'SYN Port Scan (Telnet)',
            'data': {
                'Destination Port': 23,
                'Flow Duration': 30,
                'Total Fwd Packets': 1,
                'Total Length of Fwd Packets': 54,
                'Flow Bytes/s': 1800,
                'Flow Packets/s': 33.3,
                'Fwd Packets/s': 33.3,
                'Bwd Packets/s': 0,
                'Packet Length Mean': 54,
                'Average Packet Size': 54,
                'FIN Flag Count': 0,
                'PSH Flag Count': 0,
                'ACK Flag Count': 0
            }
        },
        {
            'name': 'FIN Port Scan',
            'data': {
                'Destination Port': 3389,
                'Flow Duration': 20,
                'Total Fwd Packets': 1,
                'Total Length of Fwd Packets': 60,
                'Flow Bytes/s': 3000,
                'Flow Packets/s': 50,
                'Fwd Packets/s': 50,
                'Bwd Packets/s': 0,
                'Packet Length Mean': 60,
                'Average Packet Size': 60,
                'FIN Flag Count': 1,
                'PSH Flag Count': 0,
                'ACK Flag Count': 0
            }
        },
        {
            'name': 'High-Speed Sequential Scan',
            'data': {
                'Destination Port': 8080,
                'Flow Duration': 10,
                'Total Fwd Packets': 1,
                'Total Length of Fwd Packets': 40,
                'Flow Bytes/s': 4000,
                'Flow Packets/s': 100,
                'Fwd Packets/s': 100,
                'Bwd Packets/s': 0,
                'Packet Length Mean': 40,
                'Average Packet Size': 40,
                'FIN Flag Count': 0,
                'PSH Flag Count': 0,
                'ACK Flag Count': 0
            }
        }
    ]
    
    print("\n" + "="*70)
    print("Analyzing Test Cases:")
    print("="*70)
    
    results = []
    for test in test_cases:
        print(f"\n🌐 Test Case: {test['name']}")
        print("-" * 70)
        
        result = detector.detect(test['data'])
        results.append(result)
        
        # Display results
        print(f"  Prediction: {result['prediction']} ({result['status']})")
        print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  Severity: {result['severity']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Threat Level: {result['threat_level']}/3")
        
        if result['is_port_scan']:
            print(f"  🚨 ALERT: Port scan activity detected!")
        else:
            print(f"  ✅ Traffic appears normal")
    
    # Summary
    print("\n" + "="*70)
    print("Traffic Analysis Summary:")
    print("="*70)
    
    summary = detector.analyze_traffic_summary(results)
    print(f"  Total Flows: {summary['total_flows']}")
    print(f"  Normal: {summary['normal']}")
    print(f"  Anomalies: {summary['anomalies']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Anomaly Rate: {summary['anomaly_rate']*100:.1f}%")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    demo()
