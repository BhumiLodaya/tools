/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.jsx",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-dark': '#0B1120',
        'cyber-navy': '#161D2F',
        'cyber-cyan': '#23D5E8',
      }
    },
  },
  plugins: [],
}
