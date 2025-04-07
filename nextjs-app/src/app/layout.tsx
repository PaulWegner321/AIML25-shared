import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { UserCircle } from 'lucide-react';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ASL Translation Platform',
  description: 'Real-time ASL to text translation platform',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex h-screen">
          {/* Side Menu */}
          <div className="w-64 bg-gray-100 p-4 flex flex-col">
            <div className="mb-8">
              <h1 className="text-xl font-bold">ASL Translate</h1>
            </div>
            <nav className="flex-1">
              <ul className="space-y-2">
                <li>
                  <Link href="/" className="block p-2 hover:bg-gray-200 rounded">
                    Home
                  </Link>
                </li>
                <li>
                  <Link href="/video" className="block p-2 hover:bg-gray-200 rounded">
                    Video Demo
                  </Link>
                </li>
                <li>
                  <Link href="/text" className="block p-2 hover:bg-gray-200 rounded">
                    Text Demo
                  </Link>
                </li>
                <li>
                  <Link href="/settings" className="block p-2 hover:bg-gray-200 rounded">
                    Settings
                  </Link>
                </li>
              </ul>
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1 flex flex-col">
            {/* Top Bar with Login */}
            <div className="h-16 border-b flex items-center justify-end px-4">
              <Button variant="ghost" className="flex items-center gap-2">
                <UserCircle className="h-5 w-5" />
                <span>Sign In</span>
              </Button>
            </div>

            {/* Page Content */}
            <main className="flex-1 overflow-auto">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
} 