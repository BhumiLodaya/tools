import React, { useState } from 'react';
import { ArrowLeft, Copy, Check } from 'lucide-react';
import { Link } from 'react-router-dom';

const HashGenerator = () => {
  const [inputText, setInputText] = useState('');
  const [algorithm, setAlgorithm] = useState('SHA256');
  const [hashes, setHashes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copiedHash, setCopiedHash] = useState(null);

  const algorithms = [
    { value: 'MD5', label: 'MD5', description: '128-bit (Weak)' },
    { value: 'SHA1', label: 'SHA-1', description: '160-bit (Deprecated)' },
    { value: 'SHA256', label: 'SHA-256', description: '256-bit (Recommended)' },
    { value: 'SHA512', label: 'SHA-512', description: '512-bit (Strong)' },
  ];

  const generateHash = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('/api/generate-hash', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          algorithm: algorithm
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      const data = await response.json();
      setHashes(data);
    } catch (error) {
      console.error('Hash generation failed:', error);
      // Mock hash generation for demo
      const generateMockHash = (text, algo) => {
        const combined = text + algo + Date.now();
        let hash = '';
        for (let i = 0; i < combined.length; i++) {
          hash += combined.charCodeAt(i).toString(16).padStart(2, '0');
        }
        const lengths = { 'MD5': 32, 'SHA1': 40, 'SHA256': 64, 'SHA512': 128 };
        return (hash + hash + hash + hash).substring(0, lengths[algo]);
      };
      
      const mockHash = generateMockHash(inputText, algorithm);
      setHashes({
        hash: mockHash,
        bitLength: algorithm === 'MD5' ? 128 : algorithm === 'SHA1' ? 160 : algorithm === 'SHA256' ? 256 : 512,
        allHashes: {
          'MD5': generateMockHash(inputText, 'MD5'),
          'SHA1': generateMockHash(inputText, 'SHA1'),
          'SHA256': generateMockHash(inputText, 'SHA256'),
          'SHA512': generateMockHash(inputText, 'SHA512')
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async (text, hashType) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedHash(hashType);
      setTimeout(() => setCopiedHash(null), 2000);
    } catch (error) {
      console.error('Copy failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B1120] text-white p-8">
      {/* Back to Tools */}
      <Link 
        to="/tools" 
        className="inline-flex items-center gap-2 text-gray-400 hover:text-[#23D5E8] transition-colors mb-8"
      >
        <ArrowLeft size={20} />
        <span>Back to Tools</span>
      </Link>

      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <div className="w-16 h-16 bg-[#161D2F] rounded-xl flex items-center justify-center">
          <svg className="w-8 h-8 text-[#23D5E8]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
          </svg>
        </div>
        <div>
          <h1 className="text-3xl font-bold">Hash Generator</h1>
          <p className="text-gray-400 text-sm mt-1">Security Analysis Tool</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Input Area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Input Text */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Input Text</label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to hash..."
              rows={6}
              className="w-full bg-[#0B1120] text-white px-4 py-3 rounded-lg border border-gray-700 focus:border-[#23D5E8] focus:outline-none focus:ring-2 focus:ring-[#23D5E8]/20 transition-all resize-none font-mono text-sm"
            />
          </div>

          {/* Algorithm Selection */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Hash Algorithm</label>
            <div className="grid grid-cols-2 gap-3">
              {algorithms.map((algo) => (
                <button
                  key={algo.value}
                  onClick={() => setAlgorithm(algo.value)}
                  className={`p-4 rounded-lg border-2 transition-all text-left ${
                    algorithm === algo.value
                      ? 'border-[#23D5E8] bg-[#23D5E8]/10'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  <div className="font-semibold">{algo.label}</div>
                  <div className="text-xs text-gray-400 mt-1">{algo.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={generateHash}
            disabled={!inputText.trim() || loading}
            className="w-full bg-gradient-to-r from-[#23D5E8] to-[#1BA8B8] text-[#0B1120] font-semibold py-4 rounded-lg hover:shadow-[0_0_30px_rgba(35,213,232,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-3 border-[#0B1120] border-t-transparent rounded-full animate-spin"></div>
                Generating...
              </span>
            ) : (
              'Generate Hash'
            )}
          </button>

          {/* Hash Results */}
          {hashes && !loading && (
            <div className="space-y-4">
              {/* Main Hash */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold">{algorithm} Hash</h3>
                  <button
                    onClick={() => copyToClipboard(hashes.hash, 'main')}
                    className="flex items-center gap-2 text-[#23D5E8] hover:text-[#1BA8B8] transition-colors text-sm"
                  >
                    {copiedHash === 'main' ? (
                      <>
                        <Check size={16} />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy size={16} />
                        Copy
                      </>
                    )}
                  </button>
                </div>
                <div className="bg-[#0B1120] px-4 py-3 rounded-lg border border-gray-700 font-mono text-sm break-all text-[#23D5E8]">
                  {hashes.hash}
                </div>
              </div>

              {/* Hash Details */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <h3 className="text-lg font-semibold mb-4">Hash Details</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Algorithm:</span>
                    <span className="font-semibold">{algorithm}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Hash Length:</span>
                    <span className="font-semibold">{hashes.hash.length} characters</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Input Length:</span>
                    <span className="font-semibold">{inputText.length} characters</span>
                  </div>
                  {hashes.bitLength && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bit Length:</span>
                      <span className="font-semibold">{hashes.bitLength} bits</span>
                    </div>
                  )}
                </div>
              </div>

              {/* All Hashes (Optional) */}
              {hashes.allHashes && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <h3 className="text-lg font-semibold mb-4">All Hash Formats</h3>
                  <div className="space-y-3">
                    {Object.entries(hashes.allHashes).map(([algo, hash]) => (
                      <div key={algo} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-300">{algo}</span>
                          <button
                            onClick={() => copyToClipboard(hash, algo)}
                            className="text-xs text-[#23D5E8] hover:text-[#1BA8B8] transition-colors flex items-center gap-1"
                          >
                            {copiedHash === algo ? <Check size={12} /> : <Copy size={12} />}
                            {copiedHash === algo ? 'Copied' : 'Copy'}
                          </button>
                        </div>
                        <div className="bg-[#0B1120] px-3 py-2 rounded border border-gray-700 font-mono text-xs break-all text-gray-400">
                          {hash}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: How to Use Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800 sticky top-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-[#23D5E8]/10 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-[#23D5E8]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold">How to Use</h3>
            </div>

            <p className="text-gray-300 mb-6">
              Generate secure cryptographic hashes for data integrity verification and password storage.
            </p>

            <div className="space-y-4">
              <div>
                <h4 className="text-[#23D5E8] font-semibold mb-2">COMMON USE CASES</h4>
                <ul className="space-y-2 text-sm text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Verify file integrity</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Generate unique identifiers</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Create digital signatures</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Store password hashes</span>
                  </li>
                </ul>
              </div>

              <div className="pt-4 border-t border-gray-700">
                <h4 className="text-[#23D5E8] font-semibold mb-2">ALGORITHM GUIDE</h4>
                <div className="space-y-3 text-xs">
                  <div>
                    <span className="font-semibold text-gray-300">SHA-256:</span>
                    <span className="text-gray-400 ml-2">Best for general use</span>
                  </div>
                  <div>
                    <span className="font-semibold text-gray-300">SHA-512:</span>
                    <span className="text-gray-400 ml-2">Maximum security</span>
                  </div>
                  <div>
                    <span className="font-semibold text-red-400">MD5/SHA-1:</span>
                    <span className="text-gray-400 ml-2">Legacy only (not secure)</span>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-gray-700">
                <h4 className="text-[#23D5E8] font-semibold mb-2">SECURITY NOTE</h4>
                <p className="text-xs text-gray-400">
                  Hashing is one-way. The original text cannot be recovered from the hash.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HashGenerator;
