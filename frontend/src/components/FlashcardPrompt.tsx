'use client';

import React, { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { API_ENDPOINTS } from '@/utils/api';
import { SignEvaluationHandler } from '@/types/evaluation';

interface FlashcardPromptProps {
  onSignCaptured: SignEvaluationHandler;
  onCardChange: () => void;
}

const FlashcardPrompt = ({ onSignCaptured, onCardChange }: FlashcardPromptProps) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [currentSignIndex, setCurrentSignIndex] = useState(0);
  const [selectedModel, setSelectedModel] = useState('model1');
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isClient, setIsClient] = useState(false);

  // Define all possible signs (A-Z)
  const allSigns = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)); // A-Z

  const currentSign = allSigns[currentSignIndex];

  // Model options
  const modelOptions = [
    { id: 'model1', name: 'CNN Model 1 (Current)' },
    { id: 'model2', name: 'CNN Model 2 (Future)' },
    { id: 'gpt4o', name: 'GPT-4 Vision (Best Performance)' }
  ];

  // Initialize component
  useEffect(() => {
    setIsClient(true);
    setIsInitialized(true);
    return () => {
      stopCamera();
    };
  }, []);

  // Effect to handle camera state changes
  useEffect(() => {
    if (isCameraActive && videoRef.current) {
      const playVideo = async () => {
        try {
          await videoRef.current?.play();
        } catch (error) {
          console.error('Error playing video from useEffect:', error);
        }
      };
      playVideo();
    }
  }, [isCameraActive]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        setCameraError(null);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError('Could not access camera');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  const captureSign = async () => {
    try {
      if (!videoRef.current || !canvasRef.current) {
        console.error('Video or canvas reference not available');
        return;
      }

      // Get the canvas context
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      if (!context) {
        console.error('Could not get canvas context');
        return;
      }

      // Set canvas dimensions to match video
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      // Draw the current video frame
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      // Convert the canvas to a blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => {
          if (b) resolve(b);
        }, 'image/jpeg', 0.95);
      });

      // Get image data for display
      const imageData = canvas.toDataURL('image/jpeg');
      setCapturedImage(imageData);
      stopCamera();

      if (blob) {
        let result;
        
        // Handle different model types
        if (selectedModel === 'gpt4o') {
          // Call GPT-4o endpoint
          const formData = new FormData();
          formData.append('file', blob, 'webcam.jpg');
          formData.append('expected_sign', currentSign);
          
          try {
            const response = await fetch(API_ENDPOINTS.evaluateGPT4o, {
              method: 'POST',
              body: formData,
            });
            
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            result = await response.json();
          } catch (error) {
            console.error('Error calling GPT-4o API:', error);
            setCameraError(`API error: ${error instanceof Error ? error.message : String(error)}`);
          }
        } else {
          // Regular CNN model evaluation
          const formData = new FormData();
          formData.append('file', blob, 'webcam.jpg');
          formData.append('model_id', selectedModel);
          formData.append('model_type', 'image_processing');
          formData.append('expected_sign', currentSign);
          
          const response = await fetch(API_ENDPOINTS.evaluateSign, {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          result = await response.json();
        }
        
        // Call the parent component's callback with the result
        onSignCaptured(imageData, currentSign, result);
      }
    } catch (error) {
      console.error('Error capturing sign:', error);
      setCameraError('Failed to capture sign');
    }
  };

  const handleTryAgain = () => {
    setCapturedImage(null);
    setCameraError(null);
    startCamera();
  };

  const handleNextCard = () => {
    const nextIndex = (currentSignIndex + 1) % allSigns.length;
    setCurrentSignIndex(nextIndex);
    setCapturedImage(null);
    onCardChange();
  };

  const handlePreviousCard = () => {
    const prevIndex = (currentSignIndex - 1 + allSigns.length) % allSigns.length;
    setCurrentSignIndex(prevIndex);
    setCapturedImage(null);
    onCardChange();
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Model Selector Dropdown */}
      <div className="mb-4">
        <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-1">
          Select Model
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="block w-full px-3 py-1.5 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm"
        >
          {modelOptions.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </div>

      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-2">Sign the Letter/Number</h2>
        <div className="text-6xl font-bold text-blue-600">{currentSign}</div>
        <div className="text-sm text-gray-500 mt-2">
          Card {currentSignIndex + 1} of {allSigns.length}
        </div>
      </div>

      <div className="relative w-full h-96 bg-gray-100 rounded-lg overflow-hidden mb-4">
        {isInitialized ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              controls={false}
              className={`w-full h-full object-cover ${isCameraActive ? '' : 'hidden'}`}
            />
            {!isCameraActive && (
              <div className="flex items-center justify-center h-full">
                <p className="text-gray-500">Camera is off</p>
              </div>
            )}
          </>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500">Loading camera...</p>
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {cameraError && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
          {cameraError}
        </div>
      )}

      <div className="flex justify-center space-x-4 mb-4">
        {capturedImage ? (
          <button
            onClick={handleTryAgain}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        ) : !isCameraActive ? (
          <button
            onClick={startCamera}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Start Camera
          </button>
        ) : (
          <>
            <button
              onClick={captureSign}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Capture Sign
            </button>
            <button
              onClick={stopCamera}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Stop Camera
            </button>
          </>
        )}
      </div>

      <div className="flex justify-center space-x-4">
        <button
          onClick={handlePreviousCard}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          Previous
        </button>
        <button
          onClick={handleNextCard}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default FlashcardPrompt;