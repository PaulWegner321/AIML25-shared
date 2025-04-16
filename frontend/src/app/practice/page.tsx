'use client';

import { useState } from 'react';
import FlashcardPrompt from '@/components/FlashcardPrompt';
import FeedbackBox from '@/components/FeedbackBox';
import { API_ENDPOINTS } from '@/utils/api';

export default function PracticePage() {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);

  const handleSignEvaluation = async (imageData: string) => {
    try {
      const response = await fetch(API_ENDPOINTS.evaluateSign, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          expected_sign: 'A', // TODO: Get from flashcard
        }),
      });

      const data = await response.json();
      setFeedback(data.feedback);
      setIsCorrect(data.is_correct);
    } catch (error) {
      console.error('Error evaluating sign:', error);
      setFeedback('Error evaluating sign. Please try again.');
      setIsCorrect(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Practice ASL Signs</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <FlashcardPrompt
            currentSign="A"
            onSignCaptured={handleSignEvaluation}
          />
        </div>
        
        <div>
          <FeedbackBox
            feedback={feedback}
            isCorrect={isCorrect}
          />
        </div>
      </div>
    </div>
  );
} 