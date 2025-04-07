'use client';

import { useState, useRef, useEffect } from 'react';

export default function LiveTranslate() {
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedText, setTranslatedText] = useState('');
  const [qualityScore, setQualityScore] = useState<number | null>(null);
  const [suggestions, setSuggestions] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream;
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const startTranslation = async () => {
    setIsTranslating(true);
    
    // Mock ASL tokens for demonstration
    const mockTokens = ['HELLO', 'HOW', 'ARE', 'YOU'];
    
    try {
      // Call the translation API
      const translateResponse = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tokens: mockTokens }),
      });
      
      const translateData = await translateResponse.json();
      setTranslatedText(translateData.translated_text);
      
      // Call the quality assessment API
      const judgeResponse = await fetch('/api/judge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          tokens: mockTokens, 
          translated_text: translateData.translated_text 
        }),
      });
      
      const judgeData = await judgeResponse.json();
      setQualityScore(judgeData.score);
      setSuggestions(judgeData.suggestions);
    } catch (error) {
      console.error('Error during translation:', error);
      setTranslatedText('Error occurred during translation');
    }
  };

  const stopTranslation = () => {
    setIsTranslating(false);
    setTranslatedText('');
    setQualityScore(null);
    setSuggestions(null);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Live ASL Translation</h1>
      
      <div className="grid grid-cols-1 gap-8">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
          </div>
          
          <div className="mt-4 flex justify-center gap-4">
            <button
              onClick={startCamera}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Start Camera
            </button>
            
            {!isTranslating ? (
              <button
                onClick={startTranslation}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                Start Translation
              </button>
            ) : (
              <button
                onClick={stopTranslation}
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
              >
                Stop Translation
              </button>
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Translation</h2>
          <div className="min-h-[100px] p-4 bg-gray-50 rounded-lg">
            {translatedText || 'Translation will appear here...'}
          </div>
          
          {qualityScore !== null && (
            <div className="mt-4">
              <h3 className="text-lg font-medium mb-2">Quality Assessment</h3>
              <div className="flex items-center mb-2">
                <span className="mr-2">Score:</span>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-blue-600 h-2.5 rounded-full" 
                    style={{ width: `${qualityScore * 10}%` }}
                  ></div>
                </div>
                <span className="ml-2">{qualityScore}/10</span>
              </div>
              {suggestions && (
                <p className="text-sm text-gray-600">{suggestions}</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
