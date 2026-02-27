import React, { useState } from 'react';
import { ArrowLeft, Globe, Copy, Check } from 'lucide-react';
import { Link } from 'react-router-dom';

const DNSLookup = () => {
  const [domain, setDomain] = useState('');
  const [loading, setLoading] = useState(false);
  const [dnsData, setDnsData] = useState(null);
  const [copiedRecord, setCopiedRecord] = useState(null);

  const lookupDNS = async () => {
    if (!domain.trim()) return;

    setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('/api/dns-lookup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain: domain }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      const data = await response.json();
      setDnsData(data);
    } catch (error) {
      console.error('DNS lookup failed:', error);
      // Mock DNS data for demo
      setDnsData({
        records: {
          A: ['93.184.216.34', '93.184.216.35'],
          AAAA: ['2606:2800:220:1:248:1893:25c8:1946'],
          MX: [{ exchange: 'mail.example.com', priority: 10 }, { exchange: 'mail2.example.com', priority: 20 }],
          TXT: ['v=spf1 include:_spf.example.com ~all', 'v=DMARC1; p=none; rua=mailto:dmarc@example.com'],
          NS: ['ns1.example.com', 'ns2.example.com']
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async (text, recordType) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedRecord(recordType);
      setTimeout(() => setCopiedRecord(null), 2000);
    } catch (error) {
      console.error('Copy failed:', error);
    }
  };

  const getRecordIcon = (type) => {
    const icons = {
      A: '🌐',
      AAAA: '🌐',
      MX: '📧',
      TXT: '📝',
      NS: '🔧',
      CNAME: '🔗',
    };
    return icons[type] || '📋';
  };

  const getRecordColor = (type) => {
    const colors = {
      A: 'text-blue-400',
      AAAA: 'text-purple-400',
      MX: 'text-green-400',
      TXT: 'text-yellow-400',
      NS: 'text-orange-400',
      CNAME: 'text-cyan-400',
    };
    return colors[type] || 'text-gray-400';
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
          <Globe className="w-8 h-8 text-[#23D5E8]" />
        </div>
        <div>
          <h1 className="text-3xl font-bold">DNS Lookup</h1>
          <p className="text-gray-400 text-sm mt-1">Security Analysis Tool</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Input Area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Domain Input */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Domain Name</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && lookupDNS()}
                placeholder="Enter domain (e.g., example.com)..."
                className="flex-1 bg-[#0B1120] text-white px-4 py-3 rounded-lg border border-gray-700 focus:border-[#23D5E8] focus:outline-none focus:ring-2 focus:ring-[#23D5E8]/20 transition-all"
              />
              <button
                onClick={lookupDNS}
                disabled={!domain.trim() || loading}
                className="px-8 bg-gradient-to-r from-[#23D5E8] to-[#1BA8B8] text-[#0B1120] font-semibold rounded-lg hover:shadow-[0_0_30px_rgba(35,213,232,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none whitespace-nowrap"
              >
                {loading ? (
                  <div className="w-5 h-5 border-3 border-[#0B1120] border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  'Lookup'
                )}
              </button>
            </div>
          </div>

          {/* DNS Results */}
          {dnsData && !loading && (
            <div className="space-y-4">
              {/* Summary */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-[#23D5E8]">
                    {Object.keys(dnsData.records || {}).length}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">Record Types Found</div>
                </div>
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-green-400">
                    {Object.values(dnsData.records || {}).flat().length}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">Total Records</div>
                </div>
              </div>

              {/* A Records */}
              {dnsData.records?.A && dnsData.records.A.length > 0 && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{getRecordIcon('A')}</span>
                      <div>
                        <h3 className={`text-lg font-semibold ${getRecordColor('A')}`}>A Records</h3>
                        <p className="text-xs text-gray-400">IPv4 Addresses</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {dnsData.records.A.map((record, index) => (
                      <div key={index} className="flex items-center justify-between bg-[#0B1120] px-4 py-3 rounded-lg border border-gray-700">
                        <span className="font-mono text-[#23D5E8]">{record}</span>
                        <button
                          onClick={() => copyToClipboard(record, `A-${index}`)}
                          className="text-gray-400 hover:text-[#23D5E8] transition-colors"
                        >
                          {copiedRecord === `A-${index}` ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* AAAA Records */}
              {dnsData.records?.AAAA && dnsData.records.AAAA.length > 0 && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{getRecordIcon('AAAA')}</span>
                      <div>
                        <h3 className={`text-lg font-semibold ${getRecordColor('AAAA')}`}>AAAA Records</h3>
                        <p className="text-xs text-gray-400">IPv6 Addresses</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {dnsData.records.AAAA.map((record, index) => (
                      <div key={index} className="flex items-center justify-between bg-[#0B1120] px-4 py-3 rounded-lg border border-gray-700">
                        <span className="font-mono text-sm text-[#23D5E8]">{record}</span>
                        <button
                          onClick={() => copyToClipboard(record, `AAAA-${index}`)}
                          className="text-gray-400 hover:text-[#23D5E8] transition-colors"
                        >
                          {copiedRecord === `AAAA-${index}` ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* MX Records */}
              {dnsData.records?.MX && dnsData.records.MX.length > 0 && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{getRecordIcon('MX')}</span>
                      <div>
                        <h3 className={`text-lg font-semibold ${getRecordColor('MX')}`}>MX Records</h3>
                        <p className="text-xs text-gray-400">Mail Servers</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {dnsData.records.MX.map((record, index) => (
                      <div key={index} className="bg-[#0B1120] px-4 py-3 rounded-lg border border-gray-700">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="text-xs font-semibold text-gray-500 bg-gray-800 px-2 py-1 rounded">
                              Priority: {record.priority || index + 1}
                            </span>
                            <span className="font-mono text-sm text-[#23D5E8]">{record.exchange || record}</span>
                          </div>
                          <button
                            onClick={() => copyToClipboard(record.exchange || record, `MX-${index}`)}
                            className="text-gray-400 hover:text-[#23D5E8] transition-colors"
                          >
                            {copiedRecord === `MX-${index}` ? <Check size={16} /> : <Copy size={16} />}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TXT Records */}
              {dnsData.records?.TXT && dnsData.records.TXT.length > 0 && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{getRecordIcon('TXT')}</span>
                      <div>
                        <h3 className={`text-lg font-semibold ${getRecordColor('TXT')}`}>TXT Records</h3>
                        <p className="text-xs text-gray-400">Text Information</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {dnsData.records.TXT.map((record, index) => (
                      <div key={index} className="bg-[#0B1120] px-4 py-3 rounded-lg border border-gray-700">
                        <div className="flex items-start justify-between gap-3">
                          <span className="font-mono text-xs text-gray-300 break-all">{record}</span>
                          <button
                            onClick={() => copyToClipboard(record, `TXT-${index}`)}
                            className="text-gray-400 hover:text-[#23D5E8] transition-colors flex-shrink-0"
                          >
                            {copiedRecord === `TXT-${index}` ? <Check size={16} /> : <Copy size={16} />}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* NS Records */}
              {dnsData.records?.NS && dnsData.records.NS.length > 0 && (
                <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{getRecordIcon('NS')}</span>
                      <div>
                        <h3 className={`text-lg font-semibold ${getRecordColor('NS')}`}>NS Records</h3>
                        <p className="text-xs text-gray-400">Name Servers</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {dnsData.records.NS.map((record, index) => (
                      <div key={index} className="flex items-center justify-between bg-[#0B1120] px-4 py-3 rounded-lg border border-gray-700">
                        <span className="font-mono text-sm text-[#23D5E8]">{record}</span>
                        <button
                          onClick={() => copyToClipboard(record, `NS-${index}`)}
                          className="text-gray-400 hover:text-[#23D5E8] transition-colors"
                        >
                          {copiedRecord === `NS-${index}` ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* No Records Found */}
              {(!dnsData.records || Object.keys(dnsData.records).length === 0) && (
                <div className="bg-[#161D2F] rounded-xl p-8 border border-gray-800 text-center">
                  <p className="text-gray-400">No DNS records found for this domain</p>
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
                Resolve domain names and view DNS records instantly for troubleshooting and security analysis.
              </p>
            </div>

            <div>
              <h4 className="text-[#23D5E8] font-semibold mb-3">RECORD TYPES</h4>
              <div className="space-y-3 text-sm">
                <div>
                  <span className="font-semibold text-blue-400">A Record:</span>
                  <p className="text-gray-400 mt-1">Maps domain to IPv4 address</p>
                </div>
                <div>
                  <span className="font-semibold text-purple-400">AAAA Record:</span>
                  <p className="text-gray-400 mt-1">Maps domain to IPv6 address</p>
                </div>
                <div>
                  <span className="font-semibold text-green-400">MX Record:</span>
                  <p className="text-gray-400 mt-1">Specifies mail servers</p>
                </div>
                <div>
                  <span className="font-semibold text-yellow-400">TXT Record:</span>
                  <p className="text-gray-400 mt-1">Contains text information (SPF, DKIM, etc.)</p>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <h4 className="text-[#23D5E8] font-semibold mb-3">COMMON USES</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li className="flex items-start gap-2">
                  <span className="text-[#23D5E8] mt-1">•</span>
                  <span>Verify domain configuration</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#23D5E8] mt-1">•</span>
                  <span>Troubleshoot email delivery</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#23D5E8] mt-1">•</span>
                  <span>Check DNS propagation</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#23D5E8] mt-1">•</span>
                  <span>Security auditing</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DNSLookup;
