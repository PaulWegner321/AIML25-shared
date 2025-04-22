import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import Footer from "@/components/Footer";
import BodyWrapper from "@/components/BodyWrapper";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ASL Learning Platform",
  description: "Learn American Sign Language through interactive practice and lookup",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <BodyWrapper className={inter.className}>
        <div className="flex min-h-screen flex-col">
          <div className="flex flex-1">
            <Sidebar />
            <main className="flex-1 ml-64 p-8">
              {children}
            </main>
          </div>
          <Footer />
        </div>
      </BodyWrapper>
    </html>
  );
}
