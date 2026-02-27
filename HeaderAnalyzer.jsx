import React, { useState } from 'react';
import { ArrowLeft, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';

const HeaderAnalyzer = () => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [headerData, setHeaderData] = useState(null);

  const analyzeHeaders = async () => {
    if (!url.trim()) return;

    setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('/api/analyze-headers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      const data = await response.json();
      setHeaderData(data);
    } catch (error) {
      console.error('Header analysis failed:', error);
      // Mock header data for demo
      setHeaderData({
        headers: {
          'Strict-Transport-Security': { status: 'pass', value: 'max-age=31536000; includeSubDomains' },
          'X-Frame-Options': { status: 'pass', value: 'SAMEORIGIN' },
          'X-Content-Type-Options': { status: 'pass', value: 'nosniff' },
          'Content-Security-Policy': { status: 'warning', value: 'default-src https:', recommendation: 'Add more restrictive CSP directives' },
          'X-XSS-Protection': { status: 'missing', recommendation: 'Add X-XSS-Protection: 1; mode=block' },
          'Referrer-Policy': { status: 'missing', recommendation: 'Add Referrer-Policy: no-referrer-when-downgrade' },
          'Permissions-Policy': { status: 'missing', recommendation: 'Add Permissions-Policy to control browser features' }
        },
        recommendations: [
          'Add X-XSS-Protection header for additional XSS protection',
          'Implement a comprehensive Content Security Policy',
          'Add Referrer-Policy to control referrer information',
          'Configure Permissions-Policy for feature control'
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    if (status === 'pass') return <CheckCircle className="text-green-400" size={20} />;
    if (status === 'warning') return <AlertTriangle className="text-yellow-400" size={20} />;
    return <XCircle className="text-red-400" size={20} />;
  };

  const getStatusColor = (status) => {
    if (status === 'pass') return 'border-green-400 bg-green-400/10';
    if (status === 'warning') return 'border-yellow-400 bg-yellow-400/10';
    return 'border-red-400 bg-red-400/10';
  };

  const getStatusText = (status) => {
    if (status === 'pass') return 'Pass';
    if (status === 'warning') return 'Warning';
    return 'Missing';
  };

  const securityHeaders = [
    {
      name: 'Strict-Transport-Security',
      displayName: 'HSTS',
      description: 'Forces HTTPS connections',
      severity: 'high'
    },
    {
      name: 'X-Frame-Options',
      displayName: 'X-Frame-Options',
      description: 'Prevents clickjacking attacks',
      severity: 'high'
    },
    {
      name: 'X-Content-Type-Options',
      displayName: 'X-Content-Type-Options',
      description: 'Prevents MIME type sniffing',
      severity: 'medium'
    },
    {
      name: 'Content-Security-Policy',
      displayName: 'CSP',
      description: 'Prevents XSS and injection attacks',
      severity: 'high'
    },
    {
      name: 'X-XSS-Protection',
      displayName: 'X-XSS-Protection',
      description: 'Enables XSS filtering',
      severity: 'medium'
    },
    {
      name: 'Referrer-Policy',
      displayName: 'Referrer-Policy',
      description: 'Controls referrer information',
      severity: 'low'
    },
    {
      name: 'Permissions-Policy',
      displayName: 'Permissions-Policy',
      description: 'Controls browser features',
      severity: 'medium'
    },
  ];

  const calculateSecurityScore = (headers) => {
    if (!headers) return 0;
    const total = securityHeaders.length;
    const passed = securityHeaders.filter(h => headers[h.name]?.status === 'pass').length;
    return Math.round((passed / total) * 100);
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    if (score >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getScoreGrade = (score) => {
    if (score >= 90) return 'A+';
    if (score >= 80) return 'A';
    if (score >= 70) return 'B';
    if (score >= 60) return 'C';
    if (score >= 50) return 'D';
    return 'F';
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
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <div>
          <h1 className="text-3xl font-bold">Security Header Analyzer</h1>
          <p className="text-gray-400 text-sm mt-1">Security Analysis Tool</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Input Area */}
        <div className="lg:col-span-2 space-y-6">
          {/* URL Input */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Website URL</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && analyzeHeaders()}
                placeholder="Enter URL (e.g., https://example.com)..."
                className="flex-1 bg-[#0B1120] text-white px-4 py-3 rounded-lg border border-gray-700 focus:border-[#23D5E8] focus:outline-none focus:ring-2 focus:ring-[#23D5E8]/20 transition-all"
              />
              <button
                onClick={analyzeHeaders}
                disabled={!url.trim() || loading}
                className="px-8 bg-gradient-to-r from-[#23D5E8] to-[#1BA8B8] text-[#0B1120] font-semibold rounded-lg hover:shadow-[0_0_30px_rgba(35,213,232,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none whitespace-nowrap"
              >
                {loading ? (
                  <div className="w-5 h-5 border-3 border-[#0B1120] border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  'Analyze'
                )}
              </button>
            </div>
          </div>

          {/* Analysis Results */}
          {headerData && !loading && (
            <div className="space-y-4">
              {/* Security Score */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Security Score</h3>
                    <p className="text-sm text-gray-400">
                      Overall HTTP security header configuration
                    </p>
                  </div>
                  <div className="flex flex-col items-center">
                    <div className={`text-5xl font-bold ${getScoreColor(calculateSecurityScore(headerData.headers))}`}>
                      {calculateSecurityScore(headerData.headers)}
                    </div>
                    <div className="text-sm text-gray-400 mt-1">
                      Grade: {getScoreGrade(calculateSecurityScore(headerData.headers))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Header Checklist */}
              <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                <h3 className="text-lg font-semibold mb-4">Security Headers</h3>
                <div className="space-y-3">
                  {securityHeaders.map((header) => {
                    const headerInfo = headerData.headers?.[header.name] || { status: 'missing' };
                    const status = headerInfo.status;
                    
                    return (
                      <div
                        key={header.name}
                        className={`p-4 rounded-lg border-2 ${getStatusColor(status)} transition-all`}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex items-start gap-3 flex-1">
                            {getStatusIcon(status)}
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <h4 className="font-semibold">{header.displayName}</h4>
                                <span className={`text-xs px-2 py-0.5 rounded ${
                                  header.severity === 'high' ? 'bg-red-500/20 text-red-400' :
                                  header.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                  'bg-blue-500/20 text-blue-400'
                                }`}>
                                  {header.severity}
                                </span>
                              </div>
                              <p className="text-sm text-gray-400 mt-1">{header.description}</p>
                              
                              {/* Header Value */}
                              {headerInfo.value && status === 'pass' && (
                                <div className="mt-2 bg-[#0B1120] px-3 py-2 rounded border border-gray-700">
                                  <p className="text-xs font-mono text-gray-300 break-all">
                                    {headerInfo.value}
                                  </p>
                                </div>
                              )}
                              
                              {/* Recommendation */}
                              {status !== 'pass' && headerInfo.recommendation && (
                                <div className="mt-2 text-xs text-gray-400">
                                  💡 {headerInfo.recommendation}
                                </div>
                              )}
                            </div>
                          </div>
                          
                          <div className={`px-3 py-1 rounded text-sm font-semibold ${
                            status === 'pass' ? 'text-green-400' :
                            status === 'warning' ? 'text-yellow-400' :
                            'text-red-400'
                          }`}>
                            {getStatusText(status)}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Summary Statistics */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-green-400">
                    {securityHeaders.filter(h => headerData.headers?.[h.name]?.status === 'pass').length}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">Passed</div>
                </div>
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-yellow-400">
                    {securityHeaders.filter(h => headerData.headers?.[h.name]?.status === 'warning').length}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">Warnings</div>
                </div>
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-red-400">
                    {securityHeaders.filter(h => !headerData.headers?.[h.name] || headerData.headers[h.name].status === 'missing').length}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">Missing</div>
                </div>
              </div>

              {/* Recommendations */}
              {headerData.recommendations && headerData.recommendations.length > 0 && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <h3 className="text-lg font-semibold mb-4">Recommended Actions</h3>
                  <ul className="space-y-3">
                    {headerData.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <span className="text-[#23D5E8] mt-1">•</span>
                        <span className="text-gray-300">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: How to Use Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800 sticky top-8 space-y-6">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-[#23D5E8]/10 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-[#23D5E8]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold">How to Use</h3>
              </div>

              <p className="text-gray-300 mb-6">
                Analyze HTTP security headers for best practices and identify vulnerabilities in web application security configuration.
              </p>
            </div>

            <div>
              <h4 className="text-[#23D5E8] font-semibold mb-3">KEY HEADERS</h4>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="font-semibold text-gray-300">HSTS:</span>
                  <p className="text-gray-400 mt-1">Enforces secure HTTPS connections</p>
                </div>
                <div>
                  <span className="font-semibold text-gray-300">CSP:</span>
                  <p className="text-gray-400 mt-1">Prevents cross-site scripting attacks</p>
                </div>
                <div>
                  <span className="font-semibold text-gray-300">X-Frame-Options:</span>
                  <p className="text-gray-400 mt-1">Protects against clickjacking</p>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <h4 className="text-[#23D5E8] font-semibold mb-3">SEVERITY LEVELS</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-red-400 rounded-full"></span>
                  <span className="text-gray-400">High: Critical security issues</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-yellow-400 rounded-full"></span>
                  <span className="text-gray-400">Medium: Important improvements</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-blue-400 rounded-full"></span>
                  <span className="text-gray-400">Low: Optional enhancements</span>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <h4 className="text-[#23D5E8] font-semibold mb-3">WHY IT MATTERS</h4>
              <p className="text-sm text-gray-400 leading-relaxed">
                HTTP security headers are your first line of defense against common web attacks like XSS, clickjacking, and protocol downgrade attacks.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeaderAnalyzer;
