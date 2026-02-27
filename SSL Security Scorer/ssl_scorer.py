"""
SSL Security Scorer - Standalone Usage Script
Testify by Trustify - Cybersecurity Project

Load the trained model and predict SSL/TLS security scores (0-100).
"""

import joblib
import pandas as pd
import numpy as np


class SSLSecurityScorer:
    """
    SSL Security Scorer using Random Forest Regressor.
    Calculates security scores (0-100) for SSL/TLS certificates.
    """
    
    def __init__(self, model_path='ssl_scoring_model.pkl',
                 imputer_path='ssl_imputer.pkl',
                 scaler_path='ssl_scaler.pkl',
                 features_path='ssl_feature_names.pkl'):
        """Load trained model and artifacts."""
        print("Loading SSL Security Scorer...")
        
        self.model = joblib.load(model_path)
        self.imputer = joblib.load(imputer_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        print(f"Model loaded successfully with {len(self.feature_names)} features")
    
    def predict_score(self, features):
        """
        Predict SSL security score (0-100) for given certificate features.
        
        Args:
            features: Dictionary or DataFrame with SSL certificate features
            
        Returns:
            float: Security score between 0 and 100
        """
        # Convert to DataFrame if dict
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Impute and scale
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict
        score = self.model.predict(X_scaled)[0]
        
        # Ensure score is within valid range
        score = float(np.clip(score, 0, 100))
        
        return score
    
    def assess_certificate(self, features):
        """
        Assess SSL certificate and provide detailed security analysis.
        
        Args:
            features: Dictionary with SSL certificate features
            
        Returns:
            dict: Detailed assessment with score, grade, and recommendations
        """
        score = self.predict_score(features)
        
        # Determine grade
        if score >= 90:
            grade = 'A'
            status = 'Excellent'
        elif score >= 80:
            grade = 'B'
            status = 'Good'
        elif score >= 70:
            grade = 'C'
            status = 'Fair'
        elif score >= 60:
            grade = 'D'
            status = 'Poor'
        else:
            grade = 'F'
            status = 'Critical'
        
        # Generate recommendations
        recommendations = []
        
        if features.get('days_to_expiry', 365) < 30:
            recommendations.append("⚠️  Certificate expires soon - renew immediately")
        elif features.get('days_to_expiry', 365) < 90:
            recommendations.append("⏰ Certificate expires in less than 90 days")
        
        if features.get('is_self_signed', 0) == 1:
            recommendations.append("🔴 Self-signed certificate - not trusted by browsers")
        
        if features.get('cipher_bits', 256) < 128:
            recommendations.append("🔴 Weak encryption - upgrade to at least 128-bit")
        elif features.get('cipher_bits', 256) < 256:
            recommendations.append("⚠️  Consider upgrading to 256-bit encryption")
        
        if features.get('is_revoked', 0) == 1:
            recommendations.append("🔴 CRITICAL: Certificate has been revoked!")
        
        if features.get('days_to_expiry', 365) <= 0:
            recommendations.append("🔴 CRITICAL: Certificate has expired!")
        
        if not recommendations:
            recommendations.append("✅ Certificate configuration looks good")
        
        return {
            'score': score,
            'grade': grade,
            'status': status,
            'recommendations': recommendations
        }
    
    def batch_score(self, certificates_list):
        """
        Score multiple SSL certificates.
        
        Args:
            certificates_list: List of feature dictionaries or DataFrame
            
        Returns:
            list: Scores for each certificate
        """
        if isinstance(certificates_list, pd.DataFrame):
            df = certificates_list
        else:
            df = pd.DataFrame(certificates_list)
        
        scores = []
        for idx in range(len(df)):
            score = self.predict_score(df.iloc[idx:idx+1])
            scores.append(score)
        
        return scores
    
    def analyze_fleet(self, results):
        """
        Analyze a fleet of SSL certificates and provide summary.
        
        Args:
            results: List of assessment results
            
        Returns:
            dict: Fleet summary statistics
        """
        scores = [r['score'] if isinstance(r, dict) else r for r in results]
        
        grade_counts = {}
        for score in scores:
            if score >= 90:
                grade = 'A'
            elif score >= 80:
                grade = 'B'
            elif score >= 70:
                grade = 'C'
            elif score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        return {
            'total_certificates': len(scores),
            'average_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'grade_distribution': grade_counts,
            'failing': sum(1 for s in scores if s < 60),
            'excellent': sum(1 for s in scores if s >= 90)
        }


def demo():
    """Demonstration of the SSL Security Scorer."""
    print("="*70)
    print("SSL Security Scorer - Demo")
    print("Testify by Trustify - Cybersecurity Project")
    print("="*70)
    
    # Initialize scorer
    scorer = SSLSecurityScorer()
    
    # Test cases
    test_cases = [
        {
            'name': 'Strong SSL Configuration (TLS 1.3, 256-bit)',
            'features': {
                'days_to_expiry': 180,
                'is_self_signed': 0,
                'cipher_bits': 256,
                'is_revoked': 0
            }
        },
        {
            'name': 'Good SSL Configuration (Valid, 128-bit)',
            'features': {
                'days_to_expiry': 90,
                'is_self_signed': 0,
                'cipher_bits': 128,
                'is_revoked': 0
            }
        },
        {
            'name': 'Expiring Soon (20 days left)',
            'features': {
                'days_to_expiry': 20,
                'is_self_signed': 0,
                'cipher_bits': 256,
                'is_revoked': 0
            }
        },
        {
            'name': 'Self-Signed Certificate',
            'features': {
                'days_to_expiry': 365,
                'is_self_signed': 1,
                'cipher_bits': 256,
                'is_revoked': 0
            }
        },
        {
            'name': 'Expired Certificate',
            'features': {
                'days_to_expiry': -10,
                'is_self_signed': 0,
                'cipher_bits': 256,
                'is_revoked': 0
            }
        },
        {
            'name': 'Revoked Certificate',
            'features': {
                'days_to_expiry': 90,
                'is_self_signed': 0,
                'cipher_bits': 256,
                'is_revoked': 1
            }
        },
        {
            'name': 'Weak Encryption (56-bit)',
            'features': {
                'days_to_expiry': 180,
                'is_self_signed': 0,
                'cipher_bits': 56,
                'is_revoked': 0
            }
        },
        {
            'name': 'Critical Issues (Expired + Self-Signed + Weak)',
            'features': {
                'days_to_expiry': -30,
                'is_self_signed': 1,
                'cipher_bits': 56,
                'is_revoked': 0
            }
        }
    ]
    
    print("\n" + "="*70)
    print("Analyzing SSL Certificates:")
    print("="*70)
    
    results = []
    for test in test_cases:
        print(f"\n🔐 Test Case: {test['name']}")
        print("-" * 70)
        
        assessment = scorer.assess_certificate(test['features'])
        results.append(assessment)
        
        # Display results
        print(f"  Security Score: {assessment['score']:.1f}/100")
        print(f"  Grade: {assessment['grade']}")
        print(f"  Status: {assessment['status']}")
        print(f"  Recommendations:")
        for rec in assessment['recommendations']:
            print(f"    {rec}")
    
    # Fleet summary
    print("\n" + "="*70)
    print("SSL Certificate Fleet Summary:")
    print("="*70)
    
    summary = scorer.analyze_fleet(results)
    print(f"  Total Certificates: {summary['total_certificates']}")
    print(f"  Average Score: {summary['average_score']:.2f}/100")
    print(f"  Min Score: {summary['min_score']:.1f}")
    print(f"  Max Score: {summary['max_score']:.1f}")
    print(f"\n  Grade Distribution:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = summary['grade_distribution'].get(grade, 0)
        if count > 0:
            print(f"    Grade {grade}: {count} certificates")
    print(f"\n  Excellent (A): {summary['excellent']} certificates")
    print(f"  Failing (F): {summary['failing']} certificates")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    demo()
