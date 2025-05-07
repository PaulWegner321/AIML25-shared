'use client';

import Sidebar from "@/components/Sidebar";
import Footer from "@/components/Footer";
import BodyWrapper from "@/components/BodyWrapper";
import { useState } from "react";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <BodyWrapper className="">
      <div className="flex min-h-screen flex-col">
        {/* Burger Menu Button */}
        <button
          onClick={() => setIsSidebarOpen(true)}
          className="fixed top-4 left-4 z-40 md:hidden bg-gray-900 text-white p-2 rounded-lg hover:bg-gray-800"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* Overlay */}
        {isSidebarOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-20 md:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}

        <div className="flex flex-1">
          <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />
          <main className="flex-1 md:ml-64 p-8">
            {children}
          </main>
        </div>
        <Footer />
      </div>
    </BodyWrapper>
  );
} 