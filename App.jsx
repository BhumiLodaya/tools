import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Tools from './Tools';
import PasswordStrength from './PasswordStrength';
import HashGenerator from './HashGenerator';
import PortScanner from './PortScanner';
import SSLChecker from './SSLChecker';
import DNSLookup from './DNSLookup';
import HeaderAnalyzer from './HeaderAnalyzer';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Tools />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/password-strength" element={<PasswordStrength />} />
        <Route path="/hash-generator" element={<HashGenerator />} />
        <Route path="/port-scanner" element={<PortScanner />} />
        <Route path="/ssl-checker" element={<SSLChecker />} />
        <Route path="/dns-lookup" element={<DNSLookup />} />
        <Route path="/header-analyzer" element={<HeaderAnalyzer />} />
      </Routes>
    </Router>
  );
}

export default App;
