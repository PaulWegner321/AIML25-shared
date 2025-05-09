'use client';

import { useState } from 'react';
import DescriptionBox from '@/components/DescriptionBox';
import { API_ENDPOINTS } from '@/utils/api';

interface SignDescription {
  word: string;
  description: string;
  steps: string[];
  tips: string[];
}

export default function LookupPage() {
  const [word, setWord] = useState('');
  const [description, setDescription] = useState<SignDescription | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!word.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch(API_ENDPOINTS.signDescription, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ word: word.trim() }),
      });

      const data = await response.json();
      setDescription(data);
    } catch (error) {
      console.error('Error fetching sign description:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">Lookup ASL Signs</h1>

      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-4">
          <input
            type="text"
            value={word}
            onChange={(e) => setWord(e.target.value)}
            placeholder="Enter a letter (A-Z)"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {description && <DescriptionBox description={description} />}

      {/* Usage Instructions */}
      <div className="mt-12 bg-blue-50 rounded-lg p-6 border border-blue-100">
        <h2 className="text-xl font-bold mb-4 text-blue-800">How to Use the ASL Sign Lookup</h2>
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-blue-700 mb-2">1. Enter a Letter</h3>
            <p className="text-gray-700">Type a single letter (A-Z) in the search box above to look up its ASL sign.</p>
          </div>
          <div>
            <h3 className="font-semibold text-blue-700 mb-2">2. View the Description</h3>
            <p className="text-gray-700">The description will show you:</p>
            <ul className="list-disc list-inside mt-2 text-gray-600 ml-4">
              <li>A clear explanation of the sign</li>
              <li>Step-by-step instructions for forming the sign</li>
              <li>Helpful tips for proper execution</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-blue-700 mb-2">3. Practice the Sign</h3>
            <p className="text-gray-700">Follow the steps carefully and use the tips to perfect your form. Remember to:</p>
            <ul className="list-disc list-inside mt-2 text-gray-600 ml-4">
              <li>Practice in front of a mirror</li>
              <li>Pay attention to finger positions and hand orientation</li>
              <li>Take your time to master each sign before moving on</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 