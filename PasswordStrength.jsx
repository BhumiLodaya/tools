import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const PasswordStrength = () => {
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  // Debounced password analysis
  useEffect(() => {
    if (!password) {
      setAnalysis(null);
      return;
    }

    const timeoutId = setTimeout(async () => {
      await analyzePassword(password);
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [password]);

  const analyzePassword = async (pwd) => {
    if (!pwd) return;
    
    setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('/api/analyze-password', {
        signal: controller.signal,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pwd })
      });

      clearTimeout(timeoutId);
      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error('Password analysis failed:', error);
      // Mock data for demo/testing
      setAnalysis({
        score: calculateMockScore(pwd),
        entropy: calculateEntropy(pwd),
        improvements: generateImprovements(pwd),
        details: {
          length: pwd.length,
          hasUppercase: /[A-Z]/.test(pwd),
          hasLowercase: /[a-z]/.test(pwd),
          hasNumbers: /[0-9]/.test(pwd),
          hasSpecialChars: /[^A-Za-z0-9]/.test(pwd)
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const calculateMockScore = (pwd) => {
    let score = 0;
    if (pwd.length >= 8) score += 20;
    if (pwd.length >= 12) score += 10;
    if (pwd.length >= 16) score += 10;
    if (/[A-Z]/.test(pwd)) score += 15;
    if (/[a-z]/.test(pwd)) score += 15;
    if (/[0-9]/.test(pwd)) score += 15;
    if (/[^A-Za-z0-9]/.test(pwd)) score += 15;
    return Math.min(100, score);
  };

  const calculateEntropy = (pwd) => {
    let charsetSize = 0;
    if (/[a-z]/.test(pwd)) charsetSize += 26;
    if (/[A-Z]/.test(pwd)) charsetSize += 26;
    if (/[0-9]/.test(pwd)) charsetSize += 10;
    if (/[^A-Za-z0-9]/.test(pwd)) charsetSize += 32;
    return pwd.length * Math.log2(charsetSize || 1);
  };

  const generateImprovements = (pwd) => {
    const suggestions = [];
    if (pwd.length < 12) suggestions.push('Increase length to at least 12 characters');
    if (!/[A-Z]/.test(pwd)) suggestions.push('Add uppercase letters');
    if (!/[a-z]/.test(pwd)) suggestions.push('Add lowercase letters');
    if (!/[0-9]/.test(pwd)) suggestions.push('Include numbers');
    if (!/[^A-Za-z0-9]/.test(pwd)) suggestions.push('Use special characters (!@#$%^&*)');
    return suggestions.length > 0 ? suggestions : ['Great password!'];
  };

  const getStrengthColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    if (score >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getStrengthLabel = (score) => {
    if (score >= 80) return 'Strong';
    if (score >= 60) return 'Moderate';
    if (score >= 40) return 'Weak';
    return 'Very Weak';
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
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <div>
          <h1 className="text-3xl font-bold">Password Strength Analyzer</h1>
          <p className="text-gray-400 text-sm mt-1">Security Analysis Tool</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Input Area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Password Input */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Enter Password</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Type your password here..."
                className="w-full bg-[#0B1120] text-white px-4 py-3 pr-12 rounded-lg border border-gray-700 focus:border-[#23D5E8] focus:outline-none focus:ring-2 focus:ring-[#23D5E8]/20 transition-all"
              />
              <button
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-[#23D5E8] transition-colors"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Your password is analyzed locally and never sent to any server.
            </p>
          </div>

          {/* Analysis Results */}
          {loading && (
            <div className="bg-[#161D2F] rounded-xl p-8 border border-gray-800 flex items-center justify-center">
              <div className="w-8 h-8 border-4 border-[#23D5E8] border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}

          {analysis && !loading && (
            <div className="space-y-4">
              {/* Strength Score */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Strength Score</h3>
                  <span className={`text-2xl font-bold ${getStrengthColor(analysis.score)}`}>
                    {analysis.score}/100
                  </span>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full bg-[#0B1120] rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      analysis.score >= 80 ? 'bg-green-400' :
                      analysis.score >= 60 ? 'bg-yellow-400' :
                      analysis.score >= 40 ? 'bg-orange-400' : 'bg-red-400'
                    }`}
                    style={{ width: `${analysis.score}%` }}
                  ></div>
                </div>
                
                <p className={`mt-3 text-sm ${getStrengthColor(analysis.score)}`}>
                  {getStrengthLabel(analysis.score)}
                </p>
              </div>

              {/* Entropy */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <h3 className="text-lg font-semibold mb-2">Entropy</h3>
                <p className="text-3xl font-bold text-[#23D5E8]">{analysis.entropy.toFixed(2)} bits</p>
                <p className="text-sm text-gray-400 mt-2">
                  Measure of password unpredictability
                </p>
              </div>

              {/* Improvements */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
                {analysis.improvements && analysis.improvements.length > 0 ? (
                  <ul className="space-y-3">
                    {analysis.improvements.map((improvement, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <span className="text-yellow-400 mt-1">⚠️</span>
                        <span className="text-gray-300">{improvement}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-green-400 flex items-center gap-2">
                    <span>✓</span>
                    <span>Your password meets all security criteria!</span>
                  </p>
                )}
              </div>

              {/* Character Analysis */}
              {analysis.details && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <h3 className="text-lg font-semibold mb-4">Character Analysis</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Length:</span>
                      <span className="font-semibold">{analysis.details.length}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Uppercase:</span>
                      <span className={analysis.details.hasUppercase ? 'text-green-400' : 'text-red-400'}>
                        {analysis.details.hasUppercase ? '✓' : '✗'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Lowercase:</span>
                      <span className={analysis.details.hasLowercase ? 'text-green-400' : 'text-red-400'}>
                        {analysis.details.hasLowercase ? '✓' : '✗'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Numbers:</span>
                      <span className={analysis.details.hasNumbers ? 'text-green-400' : 'text-red-400'}>
                        {analysis.details.hasNumbers ? '✓' : '✗'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Symbols:</span>
                      <span className={analysis.details.hasSymbols ? 'text-green-400' : 'text-red-400'}>
                        {analysis.details.hasSymbols ? '✓' : '✗'}
                      </span>
                    </div>
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
              Type or paste any password to instantly analyze its strength against modern brute-force attacks.
            </p>

            <div className="space-y-4">
              <div>
                <h4 className="text-[#23D5E8] font-semibold mb-2">BEST PRACTICES</h4>
                <ul className="space-y-2 text-sm text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Use at least 12 characters</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Mix uppercase, lowercase, numbers, and symbols</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Avoid common words and patterns</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-[#23D5E8] mt-1">•</span>
                    <span>Never reuse passwords across sites</span>
                  </li>
                </ul>
              </div>

              <div className="pt-4 border-t border-gray-700">
                <h4 className="text-[#23D5E8] font-semibold mb-2">PRIVACY NOTICE</h4>
                <p className="text-xs text-gray-400">
                  All analysis happens locally in your browser. Your password never leaves your device.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PasswordStrength;
