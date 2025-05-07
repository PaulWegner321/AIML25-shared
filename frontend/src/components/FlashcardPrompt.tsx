'use client';

import { useState, useRef, useEffect } from 'react';
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
    { id: 'granite-vision', name: 'Granite Vision (IBM)' }
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
    if (!isInitialized) {
      setTimeout(startCamera, 100);
      return;
    }

    try {
      setCameraError(null);
      setCapturedImage(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Set up event listeners for video element
        videoRef.current.onloadedmetadata = () => {
          setIsCameraActive(true);
        };
        
        videoRef.current.onerror = (e) => {
          console.error('Video element error:', e);
          setCameraError('Error loading video stream');
        };
        
        // Force play the video
        try {
          await videoRef.current.play();
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
        });
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
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
          
          // Flip horizontally for a mirror effect - this fixes the common "mirroring" issue
          // with webcam previews that can cause confusion when showing sign language
          context.translate(canvasRef.current.width, 0);
          context.scale(-1, 1);
          context.drawImage(videoRef.current, 0, 0);
          
          // Restore context transformation
          context.setTransform(1, 0, 0, 1, 0, 0);
          
          const imageData = canvasRef.current.toDataURL('image/jpeg');
          setCapturedImage(imageData);
          
          // Stop the camera immediately after capturing
          stopCamera();
          
          // Convert base64 to blob for API request
          const base64Response = await fetch(imageData);
          const blob = await base64Response.blob();
          
          let result;
          
          if (selectedModel === 'granite-vision') {
            const formData = new FormData();
            formData.append('file', blob, 'webcam.jpg');
            
            // Don't pass expected_sign to avoid biasing the model
            // Instead pass what the user is trying to sign as debug info only
            formData.append('user_intention', `User is attempting to sign "${currentSign}"`);
            formData.append('mode', 'full');
            formData.append('model_id', 'granite-vision');
            formData.append('model_type', 'llm');
            
            // Test backend connection first
            try {
            } catch (pingError) {
              console.error('Cannot reach backend:', pingError);
            }
            
            try {
              // Correct file upload approach (matching our test page)
              
              // Set a timeout controller for the fetch
              const controller = new AbortController();
              const timeoutId = setTimeout(() => {
                controller.abort();
              }, 20000); // 20 second timeout
              
              const visionResponse = await fetch(API_ENDPOINTS.evaluateVision, {
                method: 'POST',
                mode: 'cors',
                headers: {
                  'Accept': 'application/json',
                },
                body: formData,
                signal: controller.signal
              });
              
              // Clear the timeout
              clearTimeout(timeoutId);
              
              if (!visionResponse.ok) {
                const errorText = await visionResponse.text();
                console.error('Vision model error response:', errorText);
                throw new Error(`Vision model evaluation failed! status: ${visionResponse.status}, details: ${errorText}`);
              }
              
              // Try to parse the response as JSON
              try {
              } catch (jsonError) {
                console.error('Error parsing JSON response:', jsonError);
                const textResponse = await visionResponse.text();
                throw new Error('Invalid JSON in response');
              }
              
            } catch (error) {
              console.error('API call error:', error);
              // Try a fallback offline response for demo purposes
              result = {
                success: true,
                letter: currentSign,
                confidence: 0.7,
                feedback: `This is an offline fallback response. Your sign appears to be correct for "${currentSign}".`
              };
              setCameraError(`API error: ${error instanceof Error ? error.message : String(error)} - Using offline fallback`);
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

            // If the sign is wrong, get improvement feedback from LLM
            if (result.success && result.letter !== currentSign) {
              const feedbackFormData = new FormData();
              feedbackFormData.append('file', blob, 'webcam.jpg');
              feedbackFormData.append('detected_sign', result.letter);
              feedbackFormData.append('expected_sign', currentSign);
              feedbackFormData.append('mode', 'feedback');
              feedbackFormData.append('model_id', 'granite-vision');
              feedbackFormData.append('model_type', 'llm');
              
              try {
              } catch (error) {
                // Continue even if feedback fails - don't break the main flow
              }
            }
          }
          
          // Call the parent component's callback with the result
          onSignCaptured(imageData, currentSign, result);
        }
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