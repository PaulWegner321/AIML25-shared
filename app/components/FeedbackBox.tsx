'use client';

import React from 'react';
import ReactMarkdown from 'react-markdown';

interface FeedbackBoxProps {
  feedback: string;
}

export default function FeedbackBox({ feedback }: FeedbackBoxProps) {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Feedback</h2>
      <div className="prose max-w-none">
        {feedback ? (
          <ReactMarkdown>{feedback}</ReactMarkdown>
        ) : (
          <p className="text-gray-500">No feedback available yet. Make a sign to get started!</p>
        )}
      </div>
    </div>
  );
} 