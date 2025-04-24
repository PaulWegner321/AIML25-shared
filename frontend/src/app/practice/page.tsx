'use client';

import { useState } from 'react';
import FlashcardPrompt from '@/components/FlashcardPrompt';
import FeedbackBox from '@/components/FeedbackBox';
import { SignEvaluationHandler } from '@/types/evaluation';
import Link from 'next/link';

export default function PracticePage() {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [detectedLetter, setDetectedLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [expectedLetter, setExpectedLetter] = useState<string | null>(null);

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
          const customFeedback = `The model detected that you signed "${letter}", but the expected sign was "${expectedSign}". ${feedback}`;
          setFeedback(customFeedback);
        } else {
          setFeedback(feedback);
        }
      } else {
        // Handle error case
        console.log('Setting error feedback:', result.error);
        const errorMessage = result.error || 'No hand detected. Please try again.';
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