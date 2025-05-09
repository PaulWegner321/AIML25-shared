'use client';

import React, { useEffect, useRef, useState } from 'react';
import ModelSelector from '@/components/ModelSelector';
import FeedbackBox from '@/components/FeedbackBox';
import LookupBox from '@/components/LookupBox';
import { processFrame } from '@/utils/processFrame';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedModel, setSelectedModel] = useState<string>('cnn');
  const [feedback, setFeedback] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    // Setup video stream
    async function setupVideo() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            facingMode: 'user',
          },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }

        // Set canvas dimensions to match video
        if (canvasRef.current) {
          canvasRef.current.width = 640;
          canvasRef.current.height = 480;
        }

        // Start processing frames
        processFrames();
      } catch (error) {
        console.error('Error accessing webcam:', error);
        setFeedback('Error accessing webcam. Please ensure you have granted camera permissions.');
      }
    }

    setupVideo();

    return () => {
      // Cleanup video stream
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const processFrames = async () => {
    if (isProcessing) return;

    setIsProcessing(true);
    try {
      const result = await processFrame(videoRef.current, canvasRef.current, selectedModel);
      if (result) {
        setFeedback(result);
      }
    } catch (error) {
      console.error('Error processing frame:', error);
    } finally {
      setIsProcessing(false);
      // Process next frame after a short delay
      setTimeout(processFrames, 1000);
    }
  };

  const handleModelSelect = (model: string) => {
    setSelectedModel(model);
  };

  return (
    <main className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-8 text-center">ASL Learning Assistant</h1>
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <div className="mb-4">
            <ModelSelector
              selectedModel={selectedModel}
              onModelSelect={handleModelSelect}
            />
          </div>
          <div className="relative">
            <video
              ref={videoRef}
              className="w-full rounded-lg"
              style={{ transform: 'scaleX(-1)' }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
              style={{ transform: 'scaleX(-1)' }}
            />
          </div>
        </div>
        <div className="flex-1 flex flex-col gap-4">
          <LookupBox className="mb-4" />
          <FeedbackBox feedback={feedback} />
        </div>
      </div>
    </main>
  );
} 