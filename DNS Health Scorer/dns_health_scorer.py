"""
DNS Security Health Scorer - Standalone Usage Script
Testify by Trustify - Cybersecurity Project

Load the trained model and calculate DNS Security Health Scores (0-100).
Designed for React frontend integration.
"""

import joblib
import pandas as pd
import numpy as np


class DNSHealthScorer:
    """
    DNS Security Health Scorer using Random Forest Regressor.
    Calculates security health scores (0-100) for DNS configurations.
    """
    
    def __init__(self, model_path='dns_health_model.pkl',
                 scaler_path='dns_scaler.pkl',
                 features_path='dns_feature_names.pkl'):
        """Load trained model and artifacts."""
        print("Loading DNS Security Health Scorer...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        print(f"Model loaded successfully with {len(self.feature_names)} features")
    
    def get_trustify_score(self, domain_features):
        """
        Get DNS Security Health Score for React frontend integration.
        Returns a clean integer from 0-100.
        
        Args:
            domain_features: Dictionary with DNS domain features
            
        Returns:
            int: Security health score between 0 and 100
        """
        # Convert to DataFrame
        if isinstance(domain_features, dict):
            df = pd.DataFrame([domain_features])
        else:
            df = domain_features.copy()
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        score = self.model.predict(X_scaled)[0]
        
        # Ensure score is within valid range and return as integer
        score = int(np.clip(round(score), 0, 100))
        
        return score
    
    def assess_dns_security(self, domain_features):
        """
        Assess DNS security and provide detailed analysis.
        
        Args:
            domain_features: Dictionary with DNS domain features
            
        Returns:
            dict: Detailed assessment with score, grade, and recommendations
        """
        score = self.get_trustify_score(domain_features)
        
        # Determine grade and status
        if score >= 85:
            grade = 'A'
            status = 'Excellent'
            color = 'green'
        elif score >= 70:
            grade = 'B'
            status = 'Good'
            color = 'lightgreen'
        elif score >= 55:
            grade = 'C'
            status = 'Fair'
            color = 'yellow'
        elif score >= 40:
            grade = 'D'
            status = 'Poor'
            color = 'orange'
        else:
            grade = 'F'
            status = 'Critical'
            color = 'red'
        
        # Generate recommendations
        recommendations = []
        
        if domain_features.get('has_dnssec', 0) == 0 and domain_features.get('has_ds_records', 0) == 0:
            recommendations.append({
                'severity': 'high',
                'message': 'Enable DNSSEC to protect against DNS spoofing',
                'points': '+20'
            })
        
        if domain_features.get('has_spf', 0) == 0:
            recommendations.append({
                'severity': 'medium',
                'message': 'Add SPF record to prevent email spoofing',
                'points': '+15'
            })
        
        if domain_features.get('has_dmarc', 0) == 0:
            recommendations.append({
                'severity': 'medium',
                'message': 'Add DMARC record for email authentication',
                'points': '+15'
            })
        
        if domain_features.get('record_types_count', 0) < 4:
            recommendations.append({
                'severity': 'low',
                'message': 'Add more DNS record types for complete setup',
                'points': '+5'
            })
        
        if domain_features.get('ttl_value', 0) < 3600:
            recommendations.append({
                'severity': 'low',
                'message': 'Consider increasing TTL for better stability',
                'points': '+3'
            })
        
        if domain_features.get('entropy', 3.0) > 3.5:
            recommendations.append({
                'severity': 'medium',
                'message': 'Domain name has high entropy - may appear suspicious',
                'points': 'N/A'
            })
        
        if not recommendations:
            recommendations.append({
                'severity': 'info',
                'message': '✅ DNS configuration looks excellent',
                'points': ''
            })
        
        return {
            'score': score,
            'grade': grade,
            'status': status,
            'color': color,
            'recommendations': recommendations,
            'security_level': self._get_security_level(score)
        }
    
    def _get_security_level(self, score):
        """Get security level description."""
        if score >= 85:
            return 'High - Well-protected DNS infrastructure'
        elif score >= 70:
            return 'Medium-High - Good security with minor improvements needed'
        elif score >= 55:
            return 'Medium - Adequate security, upgrades recommended'
        elif score >= 40:
            return 'Low - Significant security gaps present'
        else:
            return 'Critical - Immediate security improvements required'
    
    def batch_score(self, domains_list):
        """
        Score multiple DNS configurations.
        
        Args:
            domains_list: List of domain feature dictionaries or DataFrame
            
        Returns:
            list: Scores for each domain
        """
        if isinstance(domains_list, pd.DataFrame):
            df = domains_list
        else:
            df = pd.DataFrame(domains_list)
        
        scores = []
        for idx in range(len(df)):
            score = self.get_trustify_score(df.iloc[idx:idx+1])
            scores.append(score)
        
        return scores
    
    def analyze_fleet(self, scores):
        """
        Analyze a fleet of DNS configurations.
        
        Args:
            scores: List of health scores
            
        Returns:
            dict: Fleet summary statistics
        """
        grade_counts = {}
        for score in scores:
            if score >= 85:
                grade = 'A'
            elif score >= 70:
                grade = 'B'
            elif score >= 55:
                grade = 'C'
            elif score >= 40:
                grade = 'D'
            else:
                grade = 'F'
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        return {
            'total_domains': len(scores),
            'average_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'grade_distribution': grade_counts,
            'critical': sum(1 for s in scores if s < 40),
            'excellent': sum(1 for s in scores if s >= 85)
        }


def demo():
    """Demonstration of the DNS Security Health Scorer."""
    print("="*70)
    print("DNS Security Health Scorer - Demo")
    print("Testify by Trustify - Cybersecurity Project")
    print("="*70)
    
    # Initialize scorer
    scorer = DNSHealthScorer()
    
    # Test cases
    test_cases = [
        {
            'name': 'Enterprise Domain (Full Security)',
            'features': {
                'domain_length': 15,
                'entropy': 2.5,
                'record_types_count': 8,
                'num_subdomains': 2,
                'has_dnssec': 1,
                'has_ds_records': 1,
                'has_spf': 1,
                'has_dmarc': 1,
                'ttl_value': 86400,
                'num_digits': 0,
                'num_hyphens': 0
            }
        },
        {
            'name': 'Standard Domain (Good Security)',
            'features': {
                'domain_length': 12,
                'entropy': 2.8,
                'record_types_count': 5,
                'num_subdomains': 1,
                'has_dnssec': 1,
                'has_ds_records': 0,
                'has_spf': 1,
                'has_dmarc': 1,
                'ttl_value': 3600,
                'num_digits': 0,
                'num_hyphens': 0
            }
        },
        {
            'name': 'Basic Domain (Missing Email Security)',
            'features': {
                'domain_length': 10,
                'entropy': 2.3,
                'record_types_count': 3,
                'num_subdomains': 0,
                'has_dnssec': 1,
                'has_ds_records': 1,
                'has_spf': 0,
                'has_dmarc': 0,
                'ttl_value': 3600,
                'num_digits': 0,
                'num_hyphens': 0
            }
        },
        {
            'name': 'Minimal Domain (No Security Features)',
            'features': {
                'domain_length': 8,
                'entropy': 2.0,
                'record_types_count': 2,
                'num_subdomains': 0,
                'has_dnssec': 0,
                'has_ds_records': 0,
                'has_spf': 0,
                'has_dmarc': 0,
                'ttl_value': 300,
                'num_digits': 0,
                'num_hyphens': 0
            }
        },
        {
            'name': 'Suspicious Domain (High Entropy + Digits)',
            'features': {
                'domain_length': 35,
                'entropy': 4.2,
                'record_types_count': 1,
                'num_subdomains': 4,
                'has_dnssec': 0,
                'has_ds_records': 0,
                'has_spf': 0,
                'has_dmarc': 0,
                'ttl_value': 300,
                'num_digits': 12,
                'num_hyphens': 3
            }
        }
    ]
    
    print("\n" + "="*70)
    print("Analyzing DNS Configurations:")
    print("="*70)
    
    results = []
    for test in test_cases:
        print(f"\n🌐 Test Case: {test['name']}")
        print("-" * 70)
        
        assessment = scorer.assess_dns_security(test['features'])
        results.append(assessment)
        
        # Display results
        print(f"  Trustify Score: {assessment['score']}/100")
        print(f"  Grade: {assessment['grade']}")
        print(f"  Status: {assessment['status']}")
        print(f"  Security Level: {assessment['security_level']}")
        print(f"  Recommendations:")
        for rec in assessment['recommendations'][:3]:
            severity_icon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢',
                'info': 'ℹ️'
            }.get(rec['severity'], '•')
            points = f" ({rec['points']})" if rec['points'] else ""
            print(f"    {severity_icon} {rec['message']}{points}")
    
    # Fleet summary
    print("\n" + "="*70)
    print("DNS Fleet Summary:")
    print("="*70)
    
    scores = [r['score'] for r in results]
    summary = scorer.analyze_fleet(scores)
    print(f"  Total Domains: {summary['total_domains']}")
    print(f"  Average Score: {summary['average_score']:.1f}/100")
    print(f"  Min Score: {summary['min_score']}")
    print(f"  Max Score: {summary['max_score']}")
    print(f"\n  Grade Distribution:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = summary['grade_distribution'].get(grade, 0)
        if count > 0:
            print(f"    Grade {grade}: {count} domains")
    print(f"\n  Excellent (A): {summary['excellent']} domains")
    print(f"  Critical (F): {summary['critical']} domains")
    
    # React integration example
    print("\n" + "="*70)
    print("React Frontend Integration Example:")
    print("="*70)
    print("""
// React Component Example:
import React, { useState } from 'react';

function DNSHealthChecker() {
    const [score, setScore] = useState(null);
    
    const checkDNSHealth = async (domainFeatures) => {
        const response = await fetch('/api/trustify/dns-health', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(domainFeatures)
        });
        
        const data = await response.json();
        setScore(data.score);
        
        return data;
    };
    
    return (
        <div className="dns-health-widget">
            <h3>DNS Security Health Score</h3>
            {score !== null && (
                <div className="score-display">
                    <span className="score">{score}/100</span>
                    <div className="progress-bar">
                        <div style={{width: `${score}%`}} />
                    </div>
                </div>
            )}
        </div>
    );
}
    """)
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    demo()
