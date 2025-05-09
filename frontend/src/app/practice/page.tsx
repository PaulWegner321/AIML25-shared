'use client';

import { useState } from 'react';
import FlashcardPrompt from '@/components/FlashcardPrompt';
import FeedbackBox from '@/components/FeedbackBox';
import { SignEvaluationHandler } from '@/types/evaluation';
import Link from 'next/link';
import { API_ENDPOINTS } from '@/utils/api';

export default function PracticePage() {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [detectedLetter, setDetectedLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [expectedLetter, setExpectedLetter] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [selectedLetter, setSelectedLetter] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{
    letter: string | null;
    confidence: number | null;
    description: string | null;
    steps: string[] | null;
    tips: string[] | null;
  } | null>(null);

  const handleSignEvaluation: SignEvaluationHandler = async (imageData, expectedSign, result) => {
    try {
      console.log('Processing evaluation result:', result);
      
      // Always set the expected letter
      setExpectedLetter(expectedSign);
      
      if (result.success) {
        // Extract the values, ensuring they exist
        const letter = result.letter || 'Unknown';
        const confidence = typeof result.confidence === 'number' ? result.confidence : 0.5;
        const feedback = result.feedback || 'No feedback available';

        // Compare detected letter with expected sign
        const isCorrect = letter.toUpperCase() === expectedSign.toUpperCase();
        
        console.log(`Detected "${letter}", expected "${expectedSign}", match: ${isCorrect}`);
        console.log('Setting feedback with:', { isCorrect, letter, confidence, feedback });

        // Update state with the processed values
        setIsCorrect(isCorrect);
        setDetectedLetter(letter);
        setConfidence(confidence);
        
        // Custom feedback when the detected sign doesn't match the expected sign
        if (!isCorrect) {
          const customFeedback = `The model detected that you signed "${letter}", but the expected sign was "${expectedSign}".\n\n${feedback}`;
          setFeedback(customFeedback);
        } else {
          setFeedback(feedback);
        }
      } else {
        // Handle error case
        console.log('Setting error feedback:', result.error);
        let errorMessage = 'Failed to detect sign. ';
        
        if (result.error === 'No JSON found in response') {
          errorMessage += 'The model failed to process the image. Please try again or select a different model.';
        } else {
          errorMessage += result.error || 'Please try again.';
        }
        
        setFeedback(errorMessage);
        setIsCorrect(false);
        setDetectedLetter(null);
        setConfidence(null);
      }
    } catch (error) {
      // Handle unexpected errors
      console.error('Error processing evaluation:', error);
      const errorMessage = error instanceof Error ? error.message : 'Error processing evaluation. Please try again.';
      setFeedback(errorMessage);
      setIsCorrect(false);
      setDetectedLetter(null);
      setConfidence(null);
    }
  };

  const handleCardChange = () => {
    setFeedback(null);
    setIsCorrect(null);
    setDetectedLetter(null);
    setConfidence(null);
    setExpectedLetter(null);
    console.log('Card changed, clearing feedback and state');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    if (!imageFile || !selectedLetter) {
      setError('Please select an image and a letter');
      setIsLoading(false);
      return;
    }

    try {
      // First get the prediction
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('expected_sign', selectedLetter);

      // Use the new evaluation endpoint that uses CNN → GPT-4V → Mistral pipeline
      const response = await fetch(API_ENDPOINTS.evaluate, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to evaluate sign');
      }

      const result = await response.json();
      
      if (result.success) {
        setPrediction({
          letter: result.letter,
          confidence: result.confidence,
          description: result.description,
          steps: result.steps,
          tips: result.tips
        });
      } else {
        setError(result.error || 'Failed to evaluate sign');
      }
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to process image');
    } finally {
      setIsLoading(false);
    }
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
            detectedLetter={detectedLetter}
            confidence={confidence}
            expectedLetter={expectedLetter}
          />
        </div>
      </div>
      
      <div className="mt-8 text-center">
        <Link href="/diagnostic" className="text-blue-500 hover:underline text-sm">
          View Diagnostic Images
        </Link>
        <p className="text-xs text-gray-500 mt-1">
          (For debugging and development purposes only)
        </p>
      </div>
    </div>
  );
} 