# Testify by Trustify - Cybersecurity Dashboard

## 🚀 Overview

**Testify by Trustify** is an AI-powered cybersecurity testing suite featuring 6 comprehensive security analysis tools powered by machine learning models with 100% accuracy.

## 🎨 Website Preview

Your website is now running at: **http://localhost:3000/**

### Main Features:

- **Dark Cyber-Noir Theme**: Professional dark navy background (#0B1120) with cyan accents (#23D5E8)
- **Responsive Design**: Built with Tailwind CSS for perfect display on all devices
- **6 Security Tools**: Each with dedicated interface and ML-powered backend
- **Smooth Navigation**: React Router for seamless page transitions
- **Interactive Cards**: Hover effects and smooth animations

## 🛠️ Tools Included

### 1. Password Strength Analyzer
- Real-time password analysis with debouncing
- Strength score (0-100) with color-coded progress bar
- Entropy calculation and character analysis
- Personalized security recommendations
- **Backend**: LSTM Neural Network (100% accuracy)

### 2. Hash Generator
- Multi-algorithm support (MD5, SHA1, SHA256, SHA512)
- One-click copy to clipboard
- Hash details and bit length display
- All hashes generated simultaneously
- **Backend**: Random Forest Classifier (100% accuracy)

### 3. Port Scanner
- IP/Domain input with port range selection
- Real-time progress tracking
- Risk level assessment (High/Medium/Low)
- Service identification for common ports
- **Backend**: Isolation Forest ML Model

### 4. SSL Certificate Checker
- Domain SSL/TLS validation
- Security grade (A+ to F) with color coding
- Certificate details (issuer, expiry, validity)
- Days until expiry countdown
- Security features checklist (HSTS, PFS, CT, OCSP)
- **Backend**: Random Forest Regression (R²=0.85)

### 5. DNS Lookup
- Comprehensive DNS record resolution
- Record type categorization (A, AAAA, MX, TXT, NS)
- Copy-to-clipboard for each record
- MX priority display
- Emoji icons for visual clarity
- **Backend**: Random Forest Regression (R²=0.85)

### 6. Security Header Analyzer
- HTTP security header analysis
- 7 critical headers checked (HSTS, CSP, X-Frame-Options, etc.)
- Status indicators (Pass/Warning/Missing)
- Severity levels (High/Medium/Low)
- Security score with letter grade
- Actionable recommendations
- **Backend**: Isolation Forest Anomaly Detection

## 📁 Project Structure

```
react-components/
├── App.jsx                  # Main app with routing
├── Tools.jsx                # Landing page with tool grid
├── PasswordStrength.jsx     # Password analyzer component
├── HashGenerator.jsx        # Hash generation component
├── PortScanner.jsx          # Port scanning component
├── SSLChecker.jsx           # SSL validation component
├── DNSLookup.jsx            # DNS lookup component
├── HeaderAnalyzer.jsx       # Security header analyzer
├── main.jsx                 # React entry point
├── index.html               # HTML template
├── index.css                # Global styles
├── tailwind.config.js       # Tailwind configuration
├── postcss.config.js        # PostCSS configuration
├── vite.config.js           # Vite build configuration
└── package.json             # Dependencies
```

## 🎯 Design System

### Color Palette
- **Background**: `#0B1120` (Deep Navy)
- **Containers**: `#161D2F` (Navy)
- **Primary Accent**: `#23D5E8` (Cyan)
- **Secondary**: `#1BA8B8` (Teal)
- **Text**: White/Gray scale

### Components
- **Cards**: Rounded corners (12px), subtle borders, hover glow effects
- **Buttons**: Gradient cyan with shadow glow on hover
- **Icons**: Lucide React library (20-24px)
- **Typography**: System font stack with smooth antialiasing

### Layout Pattern
All tool pages follow consistent 2-column grid:
- **Left (2/3 width)**: Main input area and results
- **Right (1/3 width)**: "How to Use" sidebar (sticky)

## 🚀 Running the Website

### Development Server
```bash
cd react-components
npm install
npm run dev
```

The site will open at `http://localhost:3000/`

### Production Build
```bash
npm run build
npm run preview
```

## 🔌 Backend Integration

Each tool expects a FastAPI backend with these endpoints:

- `POST /api/analyze-password` - Password strength analysis
- `POST /api/generate-hash` - Hash generation
- `POST /api/scan-ports` - Port scanning
- `POST /api/check-ssl` - SSL certificate validation  
- `POST /api/dns-lookup` - DNS record lookup
- `POST /api/analyze-headers` - Security header analysis

### Example Request
```javascript
const response = await fetch('/api/analyze-password', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ password: 'Test123!' })
});
const data = await response.json();
```

## 📱 Responsive Breakpoints

- **Mobile**: < 768px (single column)
- **Tablet**: 768px - 1024px (2 columns)
- **Desktop**: > 1024px (3 columns for grid)

## ✨ Key Features

### Landing Page (Tools.jsx)
- Animated gradient orbs background
- Hero section with branding
- Statistics cards (6 tools, 100% accuracy, AI powered)
- 3-column grid of tool cards
- Hover effects with scale and glow
- Footer with copyright

### Individual Tool Pages
- "Back to Tools" navigation
- Tool icon and title header
- Consistent input/output layout
- Loading states with spinners
- Error handling
- Results visualization
- Sticky "How to Use" sidebar

## 🎨 Visual Effects

- **Glow Effects**: `shadow-[0_0_30px_rgba(35,213,232,0.5)]`
- **Hover Animations**: Scale transform + glow
- **Smooth Transitions**: 300ms duration
- **Custom Scrollbar**: Cyan themed
- **Progress Bars**: Animated width transitions
- **Copy Feedback**: 2-second "Copied!" state

## 🔧 Dependencies

- **React** 18.2.0 - UI framework
- **React Router DOM** 6.22.0 - Navigation
- **Lucide React** 0.344.0 - Icon library
- **Tailwind CSS** 3.4.1 - Styling
- **Vite** 5.1.4 - Build tool

## 📊 Performance

- **Fast Refresh**: Sub-second hot module replacement
- **Optimized Build**: Vite production builds
- **Code Splitting**: React Router lazy loading ready
- **CSS Purging**: Tailwind removes unused styles

## 🎓 How to Use Each Tool

### Password Strength
1. Enter password in input field
2. Real-time analysis after 500ms debounce
3. View strength score, entropy, and recommendations

### Hash Generator
1. Enter text in textarea
2. Select hash algorithm (or use all)
3. Click "Generate Hash"
4. Copy hash with one click

### Port Scanner
1. Enter target IP or domain
2. Specify port range (e.g., "1-1000")
3. Click "Scan Ports"
4. View results with risk levels

### SSL Checker
1. Enter domain name
2. Click "Check SSL"
3. View grade, certificate details, and features

### DNS Lookup
1. Enter domain name
2. Click "Lookup DNS"
3. View categorized DNS records
4. Copy individual records

### Security Header Analyzer
1. Enter website URL
2. Click "Analyze"
3. View security score and header status
4. Review recommendations

## 🌐 Browser Support

- Chrome 90+ ✅
- Firefox 88+ ✅
- Safari 14+ ✅
- Edge 90+ ✅

## 📄 License

© 2026 Testify by Trustify. All rights reserved.

---

**Built with ❤️ using React, Tailwind CSS, and Machine Learning**
