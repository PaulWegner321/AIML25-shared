'use client';

import { useState } from 'react';
import FlashcardPrompt from '@/components/FlashcardPrompt';
import FeedbackBox from '@/components/FeedbackBox';

export default function PracticePage() {
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);

  const handleSignEvaluation = async (imageData: string, expectedSign: string, result: any) => {
    try {
      console.log('Processing evaluation result:', result);
      
      if (result.success) {
        const isCorrect = result.letter === expectedSign;
        console.log('Setting feedback with:', { isCorrect, letter: result.letter, confidence: result.confidence });
        setIsCorrect(isCorrect);

        // If there's LLM feedback, use that instead of the generic feedback
        if (result.feedback && result.feedback !== '1.') {
          setFeedback(result.feedback);
        } else {
          if (isCorrect) {
            setFeedback(`Correct! You signed '${result.letter}' with ${(result.confidence * 100).toFixed(1)}% confidence.`);
          } else {
            setFeedback(`Your sign was detected as '${result.letter}' with ${(result.confidence * 100).toFixed(1)}% confidence. Try signing '${expectedSign}' again.`);
          }
        }
      } else {
        console.log('Setting error feedback:', result.error);
        setFeedback(result.error || 'No hand detected. Please try again.');
        setIsCorrect(false);
      }
    } catch (error) {
      console.error('Error processing evaluation:', error);
      setFeedback(error instanceof Error ? error.message : 'Error processing evaluation. Please try again.');
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