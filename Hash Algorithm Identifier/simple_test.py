"""
Simple test demonstrating the predict_hash() function
Testify by Trustify - Hash Algorithm Identifier
"""

from hash_identifier import HashIdentifier

# Initialize the identifier
print("="*70)
print("Hash Algorithm Identifier - Simple Test")
print("="*70)

identifier = HashIdentifier()

# Test cases: Known hashes of "hello"
test_cases = {
    'MD5': '5d41402abc4b2a76b9719d911017c592',
    'SHA1': 'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d',
    'SHA256': '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
}

print("\nTesting hash identification (all from 'hello'):\n")

for expected_algo, hash_string in test_cases.items():
    result = identifier.predict(hash_string)
    
    print(f"{expected_algo}:")
    print(f"  Hash: {hash_string}")
    print(f"  Predicted: {result['algorithm']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  Hash Length: {result['hash_length']} chars")
    print(f"  Match: {'PASS' if result['algorithm'] == expected_algo else 'FAIL'}")
    print()

print("="*70)
print("Test completed successfully!")
print("="*70)
