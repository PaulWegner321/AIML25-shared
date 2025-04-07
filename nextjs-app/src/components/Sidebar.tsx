'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Video, Settings } from 'lucide-react';

export function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { path: '/', label: 'Live Translate', icon: Video },
    { path: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-white shadow-lg h-screen">
      <div className="p-6">
        <h1 className="text-2xl font-bold text-gray-800">ASL Translator</h1>
      </div>
      <nav className="mt-6">
        {navItems.map(({ path, label, icon: Icon }) => (
          <Link
            key={path}
            href={path}
            className={`flex items-center px-6 py-3 text-gray-700 hover:bg-gray-100 ${
              pathname === path ? 'bg-gray-100 border-r-4 border-blue-500' : ''
            }`}
          >
            <Icon className="w-5 h-5 mr-3" />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
    </aside>
  );
} 