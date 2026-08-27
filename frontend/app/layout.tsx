import type { Metadata, Viewport } from "next";
import { Inter, Newsreader, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

/* UI text — Inter, optical sizing on */
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

/* Display headlines — a journal serif, for clinical gravitas */
const newsreader = Newsreader({
  subsets: ["latin"],
  weight: ["400", "500"],
  style: ["normal", "italic"],
  variable: "--font-display",
  display: "swap",
  adjustFontFallback: false,
});

/* Measurements, codes, session IDs */
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "BeanHealth CLR — Pediatric Strabismus Screening",
  description:
    "Corneal light reflex screening tool for strabismus triage. " +
    "Detect eye misalignment in seconds using your phone camera.",
  manifest: "/manifest.json",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,   // prevent zoom-in on input focus (mobile UX)
  themeColor: "#ffffff",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="min-h-full">
      <body
        className={`${inter.variable} ${newsreader.variable} ${plexMono.variable}
                    font-sans antialiased min-h-full bg-white text-ink-900`}
      >
        {children}
      </body>
    </html>
  );
}
