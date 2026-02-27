import React, { useState } from 'react';
import { ArrowLeft, Wifi, WifiOff } from 'lucide-react';
import { Link } from 'react-router-dom';

const PortScanner = () => {
  const [target, setTarget] = useState('');
  const [portRange, setPortRange] = useState('1-1000');
  const [scanning, setScanning] = useState(false);
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState(0);

  const commonPorts = [
    { port: 80, service: 'HTTP' },
    { port: 443, service: 'HTTPS' },
    { port: 22, service: 'SSH' },
    { port: 21, service: 'FTP' },
    { port: 3306, service: 'MySQL' },
  ];

  const startScan = async () => {
    if (!target.trim()) return;

    setScanning(true);
    setProgress(0);
    setResults(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch('/api/scan-ports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          target: target,
          portRange: portRange
        })
      });

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 500);

      clearTimeout(timeoutId);
      const data = await response.json();
      
      clearInterval(progressInterval);
      setProgress(100);
      setResults(data);
    } catch (error) {
      console.error('Port scan failed:', error);
      // Mock scan results for demo
      const mockPorts = [
        { port: 80, service: 'HTTP', status: 'Open', risk: 'Low' },
        { port: 443, service: 'HTTPS', status: 'Open', risk: 'Low' },
        { port: 22, service: 'SSH', status: 'Open', risk: 'Medium' },
      ];
      setResults({
        totalScanned: 1000,
        openPorts: 3,
        closedPorts: 997,
        ports: mockPorts
      });
    } finally {
      setScanning(false);
    }
  };

  const getRiskLevel = (port) => {
    if ([3306, 5432, 1433, 27017].includes(port)) return 'High';
    if ([22, 3389, 21].includes(port)) return 'Medium';
    return 'Low';
  };

  const getRiskColor = (risk) => {
    if (risk === 'High') return 'text-red-400';
    if (risk === 'Medium') return 'text-yellow-400';
    return 'text-green-400';
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
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <div>
          <h1 className="text-3xl font-bold">Port Scanner</h1>
          <p className="text-gray-400 text-sm mt-1">Security Analysis Tool</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Input Area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Target Input */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Target IP/Domain</label>
            <input
              type="text"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="e.g., scanme.nmap.org or 192.168.1.1"
              className="w-full bg-[#0B1120] text-white px-4 py-3 rounded-lg border border-gray-700 focus:border-[#23D5E8] focus:outline-none focus:ring-2 focus:ring-[#23D5E8]/20 transition-all"
            />
          </div>

          {/* Port Range */}
          <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
            <label className="block text-sm font-medium mb-3">Port Range</label>
            <input
              type="text"
              value={portRange}
              onChange={(e) => setPortRange(e.target.value)}
              placeholder="1-1000"
              className="w-full bg-[#0B1120] text-white px-4 py-3 rounded-lg border border-gray-700 focus:border-[#23D5E8] focus:outline-none focus:ring-2 focus:ring-[#23D5E8]/20 transition-all"
            />
            <p className="text-xs text-gray-500 mt-2">
              Format: start-end (e.g., 1-1000 or 80-443)
            </p>
          </div>

          {/* Start Scan Button */}
          <button
            onClick={startScan}
            disabled={!target.trim() || scanning}
            className="w-full bg-gradient-to-r from-[#23D5E8] to-[#1BA8B8] text-[#0B1120] font-semibold py-4 rounded-lg hover:shadow-[0_0_30px_rgba(35,213,232,0.5)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
          >
            {scanning ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-3 border-[#0B1120] border-t-transparent rounded-full animate-spin"></div>
                Scanning... {progress}%
              </span>
            ) : (
              'Start Scan'
            )}
          </button>

          {/* Progress Bar */}
          {scanning && (
            <div className="bg-[#161D2F] rounded-xl p-6 border border-gray-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Scanning Progress</span>
                <span className="text-sm font-semibold text-[#23D5E8]">{progress}%</span>
              </div>
              <div className="w-full bg-[#0B1120] rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#23D5E8] to-[#1BA8B8] transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Legal Notice */}
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
            <p className="text-sm text-yellow-400">
              <strong>Note:</strong> This is a simulated scan for demonstration purposes. 
              Only scan systems you own or have explicit permission to test.
            </p>
          </div>

          {/* Scan Results */}
          {results && !scanning && (
            <div className="space-y-4">
              {/* Summary */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-[#23D5E8]">{results.totalScanned || 0}</div>
                  <div className="text-sm text-gray-400 mt-1">Ports Scanned</div>
                </div>
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-green-400">{results.openPorts || 0}</div>
                  <div className="text-sm text-gray-400 mt-1">Open Ports</div>
                </div>
                <div className="bg-[#161D2F] rounded-xl p-4 border border-gray-800">
                  <div className="text-2xl font-bold text-gray-400">{results.closedPorts || 0}</div>
                  <div className="text-sm text-gray-400 mt-1">Closed Ports</div>
                </div>
              </div>

              {/* Results Table */}
              <div className="bg-[#161D2F] rounded-xl border border-gray-800 overflow-hidden">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-lg font-semibold">Scan Results</h3>
                </div>
                
                {results.ports && results.ports.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-[#0B1120]">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                            Port
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                            Service
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                            Status
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                            Risk Level
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800">
                        {results.ports.map((portInfo, index) => {
                          const risk = getRiskLevel(portInfo.port);
                          return (
                            <tr key={index} className="hover:bg-[#0B1120]/50 transition-colors">
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className="font-mono text-[#23D5E8]">{portInfo.port}</span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className="text-gray-300">{portInfo.service || 'Unknown'}</span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                {portInfo.status === 'open' ? (
                                  <span className="flex items-center gap-2 text-green-400">
                                    <Wifi size={16} />
                                    Open
                                  </span>
                                ) : (
                                  <span className="flex items-center gap-2 text-gray-500">
                                    <WifiOff size={16} />
                                    Closed
                                  </span>
                                )}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded ${getRiskColor(risk)}`}>
                                  {risk}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-8 text-center text-gray-400">
                    No open ports found
                  </div>
                )}
              </div>
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
                Enter a target IP address or domain name to discover which ports are open and identify potential security vulnerabilities.
              </p>
            </div>

            <div>
              <h4 className="text-[#23D5E8] font-semibold mb-3">COMMON PORTS</h4>
              <div className="space-y-2">
                {commonPorts.map((item) => (
                  <div key={item.port} className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">{item.service}</span>
                    <span className="font-mono text-[#23D5E8]">{item.port}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <h4 className="text-[#23D5E8] font-semibold mb-3">RISK LEVELS</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-red-400 mt-1">●</span>
                  <span className="text-gray-400">High: Database/Admin ports exposed</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-yellow-400 mt-1">●</span>
                  <span className="text-gray-400">Medium: SSH or non-standard ports</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 mt-1">●</span>
                  <span className="text-gray-400">Low: Standard web services</span>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <h4 className="text-[#23D5E8] font-semibold mb-2">LEGAL NOTICE</h4>
              <p className="text-xs text-gray-400">
                Only scan systems you own or have explicit permission to scan. Unauthorized port scanning may be illegal.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortScanner;
