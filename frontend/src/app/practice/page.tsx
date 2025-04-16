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
      // Convert base64 image to blob
      const base64Data = imageData.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteArrays = [];
      
      for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512);
        const byteNumbers = new Array(slice.length);
        
        for (let i = 0; i < slice.length; i++) {
          byteNumbers[i] = slice.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
      }
      
      const blob = new Blob(byteArrays, { type: 'image/jpeg' });
      
      // Create FormData and append the image
      const formData = new FormData();
      formData.append('file', blob, 'sign.jpg');
      formData.append('expected_sign', 'A'); // TODO: Get from flashcard
      
      // Send the request
      const response = await fetch(API_ENDPOINTS.evaluateSign, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

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