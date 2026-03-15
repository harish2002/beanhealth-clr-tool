import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "BeanHealth — Eye Screening",
  description:
    "Corneal light reflex screening tool for strabismus triage. " +
    "Detect eye misalignment in seconds using your phone camera.",
  manifest: "/manifest.json",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,   // prevent zoom-in on input focus (mobile UX)
  themeColor: "#0a0f1a",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body
        className={`${inter.variable} font-sans antialiased h-full bg-slate-50 text-slate-900`}
      >
        {children}
      </body>
    </html>
  );
}
