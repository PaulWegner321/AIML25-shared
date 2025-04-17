"use client";

import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';

interface HandDetectorProps {
  onSignDetected?: (sign: string, confidence: number) => void;
}

const HandDetector: React.FC<HandDetectorProps> = ({ onSignDetected }) => {
  const webcamRef = useRef<Webcam>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [lastDetectedSign, setLastDetectedSign] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);

  useEffect(() => {
    const captureFrame = async () => {
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          try {
            // Convert base64 to blob
            const base64Response = await fetch(imageSrc);
            const blob = await base64Response.blob();
  
            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'webcam.jpg');
  
            // Send to backend - use local URL during development
            const response = await fetch('http://localhost:8000/evaluate-sign', {
              method: 'POST',
              body: formData,
            });
  
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
  
            const result = await response.json();
            console.log('Detection result:', result);  // Add logging
            
            if (result.success) {
              setLastDetectedSign(result.letter);
              setConfidence(result.confidence);
              if (onSignDetected) {
                onSignDetected(result.letter, result.confidence);
              }
            }
          } catch (error) {
            console.error('Error detecting sign:', error);
          }
        }
      }
    };

    let intervalId: NodeJS.Timeout;

    if (isDetecting) {
      // Capture frame every 500ms
      intervalId = setInterval(captureFrame, 500);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isDetecting, onSignDetected]);

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          className="rounded-lg"
          videoConstraints={{
            width: 640,
            height: 480,
            facingMode: "user"
          }}
        />
        {lastDetectedSign && (
          <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-4 py-2 rounded-lg">
            Detected: {lastDetectedSign} ({confidence.toFixed(2)})
          </div>
        )}
      </div>
      
      <button
        onClick={() => setIsDetecting(!isDetecting)}
        className={`px-4 py-2 rounded-lg ${
          isDetecting 
            ? 'bg-red-500 hover:bg-red-600' 
            : 'bg-blue-500 hover:bg-blue-600'
        } text-white transition-colors`}
      >
        {isDetecting ? 'Stop Detection' : 'Start Detection'}
      </button>
    </div>
  );
};

export default HandDetector; 