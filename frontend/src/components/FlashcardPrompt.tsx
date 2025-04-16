'use client';

import { useState, useRef, useEffect } from 'react';

interface FlashcardPromptProps {
  currentSign: string;
  onSignCaptured: (imageData: string) => void;
}

const FlashcardPrompt = ({ currentSign, onSignCaptured }: FlashcardPromptProps) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize component
  useEffect(() => {
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

  const captureSign = () => {
    try {
      if (videoRef.current && canvasRef.current) {
        const context = canvasRef.current.getContext('2d');
        if (context) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          context.drawImage(videoRef.current, 0, 0);
          const imageData = canvasRef.current.toDataURL('image/jpeg');
          onSignCaptured(imageData);
          console.log('Sign captured successfully');
        }
      }
    } catch (error) {
      console.error('Error capturing sign:', error);
      setCameraError('Failed to capture sign');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-2">Sign the Letter</h2>
        <div className="text-6xl font-bold text-blue-600">{currentSign}</div>
      </div>

      <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden mb-4">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          controls={false}
          className={`w-full h-full object-cover ${isCameraActive ? '' : 'hidden'}`}
          style={{ transform: 'scaleX(-1)' }}
        />
        {!isCameraActive && (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500">Camera is off</p>
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {cameraError && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
          {cameraError}
        </div>
      )}

      <div className="flex justify-center space-x-4">
        {!isCameraActive ? (
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
    </div>
  );
};

export default FlashcardPrompt; 