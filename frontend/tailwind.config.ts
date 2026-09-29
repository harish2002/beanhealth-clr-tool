import type { Config } from "tailwindcss";

/**
 * BeanHealth CLR — design tokens.
 *
 * Clinical-premium light theme: pure white ground, hairline rules, a single
 * deep-blue accent reserved for wayfinding, and ink-black for primary action.
 * Colour is never decorative here — it either encodes urgency or marks a path.
 */
const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    // URGENCY_CONFIG in lib/types.ts holds tier colour classes; without this
    // they are never generated and the URGENT/NORMAL tier blocks render blank.
    "./lib/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",

        /* Neutral ramp — cool, desaturated, calibrated for white paper */
        ink: {
          DEFAULT: "#0A1019",
          900: "#0A1019",
          800: "#16202E",
          700: "#2B3644",
          600: "#465264",
          500: "#5F6B7D",
          400: "#8A94A3",
          300: "#B4BCC8",
          200: "#D5DBE4",
          100: "#E8ECF2",
          50:  "#F5F7FA",
        },

        /* Wayfinding accent */
        clinical: {
          DEFAULT: "#1B47C4",
          700: "#16389C",
          600: "#1B47C4",
          500: "#2D5CE0",
          100: "#DCE5FB",
          50:  "#F1F5FE",
        },

        /* Method / science accent — used sparingly for measurement UI */
        vital: {
          DEFAULT: "#0E7C7B",
          600: "#0E7C7B",
          100: "#D6EDEC",
          50:  "#F0F8F8",
        },
      },

      fontFamily: {
        sans:    ["var(--font-inter)", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "Georgia", "ui-serif", "serif"],
        mono:    ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },

      letterSpacing: {
        eyebrow: "0.18em",
      },

      borderRadius: {
        card: "16px",
        field: "10px",
      },

      boxShadow: {
        /* Layered, low-opacity — paper lift rather than drop shadow */
        hairline: "0 0 0 1px rgba(10, 16, 25, 0.06)",
        card:     "0 1px 2px rgba(10, 16, 25, 0.04), 0 8px 24px -12px rgba(10, 16, 25, 0.10)",
        lift:     "0 2px 4px rgba(10, 16, 25, 0.05), 0 16px 40px -16px rgba(10, 16, 25, 0.16)",
        focus:    "0 0 0 3px rgba(27, 71, 196, 0.16)",
      },

      keyframes: {
        "rise-in": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
      },

      animation: {
        "rise-in": "rise-in 0.45s cubic-bezier(0.22, 1, 0.36, 1) both",
        "fade-in": "fade-in 0.3s ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
