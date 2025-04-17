'use client';

import { useState } from 'react';
import FlashcardPrompt from '@/components/FlashcardPrompt';
import FeedbackBox from '@/components/FeedbackBox';
import { API_ENDPOINTS } from '@/utils/api';

export default function PracticePage() {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [currentSign, setCurrentSign] = useState('A'); // Add state for current sign

  const handleSignEvaluation = async (imageData: string) => {
    try {
      console.log('Starting sign evaluation...');
      console.log('API URL:', API_ENDPOINTS.evaluateSign);
      
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
      console.log('Image converted to blob');
      
      // Create FormData and append the image
      const formData = new FormData();
      formData.append('file', blob, 'sign.jpg');
      formData.append('expected_sign', currentSign);
      console.log('FormData created with image and expected sign');
      
      // Send the request
      const response = await fetch(API_ENDPOINTS.evaluateSign, {
        method: 'POST',
        body: formData,
      });
      
      console.log('Response received:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.error('Error response:', errorData);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Evaluation result:', data);

      if (data.success) {
        const isCorrect = data.letter === currentSign;
        setIsCorrect(isCorrect);
        if (isCorrect) {
          setFeedback(`Correct! You signed '${data.letter}' with ${(data.confidence * 100).toFixed(1)}% confidence.`);
        } else {
          setFeedback(`Your sign was detected as '${data.letter}' with ${(data.confidence * 100).toFixed(1)}% confidence. Try signing '${currentSign}' again.`);
        }
      } else {
        setFeedback(data.error || 'No hand detected. Please try again.');
        setIsCorrect(false);
      }
    } catch (error) {
      console.error('Error evaluating sign:', error);
      setFeedback(error instanceof Error ? error.message : 'Error evaluating sign. Please try again.');
      setIsCorrect(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">Practice ASL Signs</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <FlashcardPrompt
            currentSign={currentSign}
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