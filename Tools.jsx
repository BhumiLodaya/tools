import React from 'react';
import { Link } from 'react-router-dom';
import { Key, Hash, Wifi, Shield, Globe, FileText } from 'lucide-react';

const Tools = () => {
  const tools = [
    {
      id: 1,
      name: 'Password Strength',
      description: 'Check if your password can withstand brute-force attacks.',
      icon: Key,
      path: '/password-strength',
      iconColor: 'text-[#23D5E8]'
    },
    {
      id: 2,
      name: 'Hash Generator',
      description: 'Generate secure cryptographic hashes for data integrity.',
      icon: Hash,
      path: '/hash-generator',
      iconColor: 'text-[#23D5E8]'
    },
    {
      id: 3,
      name: 'Port Scanner',
      description: 'Discover open ports and potential security vulnerabilities.',
      icon: Wifi,
      path: '/port-scanner',
      iconColor: 'text-[#23D5E8]'
    },
    {
      id: 4,
      name: 'SSL Checker',
      description: 'Verify SSL certificate validity and encryption strength.',
      icon: Shield,
      path: '/ssl-checker',
      iconColor: 'text-[#23D5E8]'
    },
    {
      id: 5,
      name: 'DNS Lookup',
      description: 'Resolve domain names and view DNS records instantly.',
      icon: Globe,
      path: '/dns-lookup',
      iconColor: 'text-[#23D5E8]'
    },
    {
      id: 6,
      name: 'Header Analyzer',
      description: 'Analyze HTTP security headers for best practices.',
      icon: FileText,
      path: '/header-analyzer',
      iconColor: 'text-[#23D5E8]'
    }
  ];

  return (
    <div className="min-h-screen bg-[#0B1120] text-white px-8 py-16">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-2">
            Security <span className="text-[#23D5E8]">Tools</span>
          </h1>
          <p className="text-gray-400">Comprehensive cybersecurity analysis toolkit</p>
        </div>

        {/* Tools Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tools.map((tool) => {
            const Icon = tool.icon;
            return (
              <Link
                key={tool.id}
                to={tool.path}
                className="group bg-[#161D2F] rounded-lg p-6 border border-gray-800 hover:border-[#23D5E8] transition-all duration-300"
              >
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className="flex-shrink-0">
                    <Icon className={`w-10 h-10 ${tool.iconColor}`} strokeWidth={1.5} />
                  </div>
                  
                  {/* Content */}
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold mb-2 text-white group-hover:text-[#23D5E8] transition-colors">
                      {tool.name}
                    </h3>
                    <p className="text-sm text-gray-400 leading-relaxed">
                      {tool.description}
                    </p>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Tools;
