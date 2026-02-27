"""Quick test of hash_identifier predict_hash() function"""
from hash_identifier import predict_hash

print("\n" + "="*70)
print("Testing predict_hash() function")
print("="*70)

# Test 1: MD5
print("\n1. Testing MD5 hash:")
predict_hash('5d41402abc4b2a76b9719d911017c592')

# Test 2: SHA1  
print("\n2. Testing SHA-1 hash:")
predict_hash('aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d')

# Test 3: SHA256
print("\n3. Testing SHA-256 hash:")
predict_hash('2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')

print("\n" + "="*70)
print("All tests completed!")
print("="*70)
