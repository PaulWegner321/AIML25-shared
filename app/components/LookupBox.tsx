'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface SignDescription {
  word: string;
  description: string;
  steps: string[];
  tips: string[];
}

interface LookupBoxProps {
  className?: string;
}

export default function LookupBox({ className = '' }: LookupBoxProps) {
  const [signName, setSignName] = useState('');
  const [description, setDescription] = useState<SignDescription | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLookup = async () => {
    if (!signName) return;

    setIsLoading(true);
    setError('');
    setDescription(null);

    try {
      const response = await fetch('/api/lookup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ signName }),
      });

      if (!response.ok) {
        throw new Error('Failed to get sign description');
      }

      const data = await response.json();
      setDescription(data);
    } catch (err) {
      setError('Failed to get sign description. Please try again.');
      console.error('Error in lookup:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`p-4 bg-white rounded-lg shadow ${className}`}>
      <h2 className="text-xl font-bold mb-4">ASL Sign Lookup</h2>
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={signName}
          onChange={(e) => setSignName(e.target.value.toUpperCase())}
          placeholder="Enter a letter (A-Z)"
          className="flex-1 p-2 border rounded"
          maxLength={1}
        />
        <button
          onClick={handleLookup}
          disabled={isLoading || !signName}
          className={`px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed`}
        >
          {isLoading ? 'Loading...' : 'Lookup'}
        </button>
      </div>
      {error && (
        <div className="text-red-500 mb-4">{error}</div>
      )}
      {description && (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-2">Description</h3>
            <div className="prose max-w-none">
              <ReactMarkdown>{description.description}</ReactMarkdown>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">Steps</h3>
            <ol className="list-decimal list-inside space-y-2">
              {description.steps.map((step, index) => (
                <li key={index} className="prose max-w-none">
                  <ReactMarkdown>{step}</ReactMarkdown>
                </li>
              ))}
            </ol>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">Tips</h3>
            <ul className="list-disc list-inside space-y-2">
              {description.tips.map((tip, index) => (
                <li key={index} className="prose max-w-none">
                  <ReactMarkdown>{tip}</ReactMarkdown>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
} 