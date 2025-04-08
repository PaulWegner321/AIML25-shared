'use client';

import { useState, useRef, useEffect } from 'react';

export default function CameraTest() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const toggleCamera = async () => {
    if (isCameraOn) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraOn(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: 640,
            height: 480
          } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Force play after a short delay
          setTimeout(() => {
            if (videoRef.current) {
              videoRef.current.play()
                .then(() => console.log("Video playing successfully"))
                .catch(e => console.error("Error playing video:", e));
            }
          }, 100);
        }
        
        streamRef.current = stream;
        setIsCameraOn(true);
        console.log("Camera stream obtained:", stream);
      } catch (error) {
        console.error('Error accessing camera:', error);
      }
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Camera Test</h1>
      <button 
        onClick={toggleCamera}
        className="bg-blue-500 text-white px-4 py-2 rounded mb-4"
      >
        {isCameraOn ? 'Turn Off Camera' : 'Turn On Camera'}
      </button>
      <div className="w-full max-w-2xl aspect-video bg-gray-200">
        {isCameraOn ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500">
            Camera is off
          </div>
        )}
      </div>
      <div className="mt-4 p-2 bg-gray-100 rounded">
        <p>Camera status: {isCameraOn ? 'On' : 'Off'}</p>
        <p>Stream active: {streamRef.current ? 'Yes' : 'No'}</p>
      </div>
    </div>
  );
} 