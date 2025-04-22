'use client';

import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { API_ENDPOINTS } from '@/utils/api';

interface FlashcardPromptProps {
  onSignCaptured: (imageData: string, expectedSign: string, result: any) => void;
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

  // Define all possible signs (A-Z and 0-9)
  const allSigns = [
    ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)), // A-Z
    ...Array.from({ length: 10 }, (_, i) => i.toString()) // 0-9
  ];

  const currentSign = allSigns[currentSignIndex];

  // Model options
  const modelOptions = [
    { id: 'model1', name: 'CNN Model 1 (Current)' },
    { id: 'model2', name: 'CNN Model 2 (Future)' },
    { id: 'watson', name: 'Watson LLM' }
  ];

  // Initialize component
  useEffect(() => {
    setIsClient(true);
    console.log('Component mounted, initializing...');
    setIsInitialized(true);
    return () => {
      console.log('Component unmounting, cleaning up...');
      stopCamera();
    };
  }, []);

  // Effect to handle camera state changes
  useEffect(() => {
    if (isCameraActive && videoRef.current) {
      console.log('Camera active, ensuring video is playing');
      const playVideo = async () => {
        try {
          await videoRef.current?.play();
          console.log('Video playback started from useEffect');
        } catch (error) {
          console.error('Error playing video from useEffect:', error);
        }
      };
      playVideo();
    }
  }, [isCameraActive]);

  const startCamera = async () => {
    if (!isInitialized) {
      console.log('Component not fully initialized yet, waiting...');
      setTimeout(startCamera, 100);
      return;
    }

    try {
      setCameraError(null);
      setCapturedImage(null);
      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      
      console.log('Camera access granted, stream obtained');
      
      if (videoRef.current) {
        console.log('Setting video source...');
        videoRef.current.srcObject = stream;
        
        // Set up event listeners for video element
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          setIsCameraActive(true);
          console.log('Camera started successfully');
        };
        
        videoRef.current.onerror = (e) => {
          console.error('Video element error:', e);
          setCameraError('Error loading video stream');
        };
        
        // Force play the video
        try {
          await videoRef.current.play();
          console.log('Video playback started');
        } catch (playError) {
          console.error('Error playing video:', playError);
          setCameraError('Error playing video stream');
        }
      } else {
        console.error('Video element reference is null');
        setCameraError('Video element not found');
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError(error instanceof Error ? error.message : 'Failed to access camera');
    }
  };

  const stopCamera = () => {
    try {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => {
          track.stop();
          console.log('Camera track stopped');
        });
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
      console.log('Camera stopped successfully');
    } catch (error) {
      console.error('Error stopping camera:', error);
    }
  };

  const captureSign = async () => {
    try {
      if (videoRef.current && canvasRef.current) {
        const context = canvasRef.current.getContext('2d');
        if (context) {
          // Capture the image
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          context.drawImage(videoRef.current, 0, 0);
          const imageData = canvasRef.current.toDataURL('image/jpeg');
          setCapturedImage(imageData);
          
          // Convert base64 to blob for API request
          const base64Response = await fetch(imageData);
          const blob = await base64Response.blob();
          
          let result;
          
          if (selectedModel === 'watson') {
            console.log('Using Watson LLM for evaluation...');
            // First, get the detected sign from the CNN model
            const imageFormData = new FormData();
            imageFormData.append('file', blob, 'webcam.jpg');
            imageFormData.append('model_id', 'model1');
            imageFormData.append('model_type', 'image_processing');
            
            console.log('Sending image to CNN model for sign detection...');
            const cnnResponse = await fetch(API_ENDPOINTS.evaluateSign, {
              method: 'POST',
              body: imageFormData,
            });
            
            if (!cnnResponse.ok) {
              throw new Error(`CNN evaluation failed! status: ${cnnResponse.status}`);
            }
            
            const cnnResult = await cnnResponse.json();
            console.log('CNN detection result:', cnnResult);
            
            if (cnnResult.success) {
              // Now send to LLM for evaluation
              console.log('Sending detected sign to LLM for evaluation...');
              const llmFormData = new FormData();
              llmFormData.append('file', blob, 'webcam.jpg');
              llmFormData.append('detected_sign', cnnResult.letter);
              llmFormData.append('expected_sign', currentSign);
              llmFormData.append('model_type', 'llm');
              
              const llmResponse = await fetch(API_ENDPOINTS.evaluateLLM, {
                method: 'POST',
                body: llmFormData,
              });
              
              if (!llmResponse.ok) {
                throw new Error(`LLM evaluation failed! status: ${llmResponse.status}`);
              }
              
              const llmResult = await llmResponse.json();
              console.log('LLM evaluation result:', llmResult);

              // Combine CNN detection confidence with LLM feedback
              result = {
                ...llmResult,
                confidence: cnnResult.confidence, // Use CNN's confidence as it's more relevant for sign detection
                letter: cnnResult.letter, // Use CNN's letter detection
              };
            } else {
              throw new Error('CNN failed to detect sign');
            }
          } else {
            // Regular CNN model evaluation
            console.log('Using CNN model for evaluation...');
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
            console.log('CNN evaluation result:', result);
          }
          
          // Call the parent component's callback with the result
          onSignCaptured(imageData, currentSign, result);
          stopCamera();
          console.log('Sign captured and evaluated successfully');
        }
      }
    } catch (error) {
      console.error('Error capturing sign:', error);
      setCameraError('Failed to capture sign');
    }
  };

  const handleNextCard = () => {
    const nextIndex = (currentSignIndex + 1) % allSigns.length;
    setCurrentSignIndex(nextIndex);
    setCapturedImage(null);
    onCardChange();
    console.log('Moving to next card:', allSigns[nextIndex]);
  };

  const handlePreviousCard = () => {
    const prevIndex = (currentSignIndex - 1 + allSigns.length) % allSigns.length;
    setCurrentSignIndex(prevIndex);
    setCapturedImage(null);
    onCardChange();
    console.log('Moving to previous card:', allSigns[prevIndex]);
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

      <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden mb-4">
        {capturedImage ? (
          <Image
            src={capturedImage}
            alt="Captured sign"
            fill
            className="object-cover"
          />
        ) : isClient ? (
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
            onClick={startCamera}
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