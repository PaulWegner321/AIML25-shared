'use client';

import { useState } from 'react';
import FlashcardPrompt from '@/components/FlashcardPrompt';
import FeedbackBox from '@/components/FeedbackBox';
import { API_ENDPOINTS } from '@/utils/api';

export default function PracticePage() {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);

  const handleSignEvaluation = async (imageData: string, expectedSign: string) => {
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
      formData.append('expected_sign', expectedSign);
      console.log('FormData created with image and expected sign');
      
      // Send the request
      console.log('Sending request to API...');
      const response = await fetch(API_ENDPOINTS.evaluateSign, {
        method: 'POST',
        body: formData,
      }).catch(error => {
        console.error('Fetch error:', error);
        throw error;
      });
      
      console.log('Response received:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      let responseText;
      try {
        responseText = await response.text();
        console.log('Raw response:', responseText);
      } catch (error) {
        console.error('Error reading response text:', error);
        throw error;
      }

      let data;
      try {
        data = JSON.parse(responseText);
        console.log('Parsed response data:', data);
      } catch (error) {
        console.error('Error parsing JSON:', error);
        throw new Error('Invalid JSON response from server');
      }

      if (!response.ok) {
        console.error('Error response:', data);
        throw new Error(data?.detail || `HTTP error! status: ${response.status}`);
      }

      if (data.success) {
        const isCorrect = data.letter === expectedSign;
        console.log('Setting feedback with:', { isCorrect, letter: data.letter, confidence: data.confidence });
        setIsCorrect(isCorrect);
        if (isCorrect) {
          setFeedback(`Correct! You signed '${data.letter}' with ${(data.confidence * 100).toFixed(1)}% confidence.`);
        } else {
          setFeedback(`Your sign was detected as '${data.letter}' with ${(data.confidence * 100).toFixed(1)}% confidence. Try signing '${expectedSign}' again.`);
        }
      } else {
        console.log('Setting error feedback:', data.error);
        setFeedback(data.error || 'No hand detected. Please try again.');
        setIsCorrect(false);
      }
    } catch (error) {
      console.error('Error evaluating sign:', error);
      setFeedback(error instanceof Error ? error.message : 'Error evaluating sign. Please try again.');
      setIsCorrect(false);
    }
  };

  const handleCardChange = () => {
    setFeedback('');
    setIsCorrect(false);
    console.log('Card changed, clearing feedback and state');
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8">Practice ASL Signs</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <FlashcardPrompt
            onSignCaptured={handleSignEvaluation}
            onCardChange={handleCardChange}
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