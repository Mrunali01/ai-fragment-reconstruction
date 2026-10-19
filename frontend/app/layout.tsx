import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jigsaw AI · Find the bigger picture",
  description:
    "Reconstruct a shuffled 25-piece image with a trained CNN or classical edge matching.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
