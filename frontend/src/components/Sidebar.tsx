'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar = ({ isOpen, onClose }: SidebarProps) => {
  const pathname = usePathname();

  // Close sidebar on navigation on mobile
  useEffect(() => {
    if (isOpen) {
      onClose();
    }
  }, [pathname, isOpen, onClose]);

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Practice', path: '/practice' },
    { name: 'Lookup', path: '/lookup' },
    { name: 'Settings', path: '/settings' },
  ];

  return (
    <div 
      className={`fixed top-0 left-0 h-full bg-gray-900 text-white p-4 transition-transform duration-300 ease-in-out z-30
        md:translate-x-0 md:w-64
        ${isOpen ? 'translate-x-0 w-64' : '-translate-x-full w-64'}
      `}
    >
      <div className="mb-8 flex justify-between items-center">
        <h1 className="text-2xl font-bold">ASL Learning</h1>
        <button 
          onClick={onClose}
          className="md:hidden text-white p-2 hover:bg-gray-800 rounded-lg"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      <nav>
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <Link
                href={item.path}
                className={`block px-4 py-2 rounded-lg transition-colors ${
                  pathname === item.path
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800'
                }`}
              >
                {item.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      <div className="absolute bottom-4 left-4 right-4">
        <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          Login
        </button>
      </div>
    </div>
  );
};

export default Sidebar; 