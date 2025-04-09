'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, Loader2, Camera, CameraOff } from 'lucide-react';
import { API_CONFIG } from '@/config/api';

export default function VideoDemo() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translation, setTranslation] = useState('');
  const [feedback, setFeedback] = useState('');
  const [score, setScore] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Clean up camera stream when component unmounts
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Initialize video element
  useEffect(() => {
    let mounted = true;
    
    const setupVideoElement = async () => {
      if (videoRef.current && isCameraOn) {
        const videoElement = videoRef.current;
        
        // Set initial video element properties
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;

        // If we have a stream, set it up
        if (streamRef.current) {
          try {
            videoElement.srcObject = streamRef.current;
            await videoElement.play();
          } catch (error) {
            console.error("Error playing video:", error);
          }
        }
      }
    };

    setupVideoElement();

    // Cleanup function
    return () => {
      mounted = false;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [isCameraOn]);

  const toggleCamera = async () => {
    if (isCameraOn) {
      // Turn off camera
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraOn(false);
    } else {
      // Turn on camera
      try {
        const constraints = {
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
            frameRate: { ideal: 30 }
          },
          audio: false
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        setIsCameraOn(true);
        setError(null);
      } catch (error) {
        console.error('Error accessing camera:', error);
        setError('Failed to access camera. Please make sure you have granted camera permissions.');
        setIsCameraOn(false);
      }
    }
  };

  const startTranslation = async () => {
    if (!isCameraOn) {
      setError('Please turn on the camera first.');
      return;
    }

    setIsTranslating(true);
    setError(null);

    try {
      // TODO: Implement frame capture and ASL detection
      // For now, we'll use dummy tokens
      const dummyTokens = ['hello', 'world'];
      
      // Call the translation API
      const translationResponse = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.TRANSLATE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tokens: dummyTokens }),
      });

      if (!translationResponse.ok) {
        throw new Error('Translation request failed');
      }

      const translationData = await translationResponse.json();
      setTranslation(translationData.translation);

      // Call the judge API
      const judgeResponse = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.JUDGE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          translation: translationData.translation,
          tokens: dummyTokens,
        }),
      });

      if (!judgeResponse.ok) {
        throw new Error('Judgment request failed');
      }

      const judgeData = await judgeResponse.json();
      setFeedback(judgeData.feedback);
      setScore(judgeData.score);
    } catch (error) {
      console.error('Translation error:', error);
      setError('Failed to translate video feed. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Video ASL Translation</CardTitle>
            <CardDescription>
              Use your camera to perform ASL signs and get real-time translation.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid w-full items-center gap-4">
              <div className="flex flex-col space-y-1.5">
                <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden relative">
                  {isCameraOn ? (
                    <>
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                          backgroundColor: 'black',
                          transform: 'scaleX(-1)' // Mirror the video
                        }}
                      />
                      <div className="absolute top-2 right-2 text-xs text-white bg-black/50 px-2 py-1 rounded">
                        Stream Active
                      </div>
                    </>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      Camera is off
                    </div>
                  )}
                </div>
                <div className="flex justify-center mt-4">
                  <Button
                    onClick={toggleCamera}
                    variant={isCameraOn ? "destructive" : "default"}
                    className="w-40"
                  >
                    {isCameraOn ? (
                      <>
                        <CameraOff className="mr-2 h-4 w-4" />
                        Turn Off Camera
                      </>
                    ) : (
                      <>
                        <Camera className="mr-2 h-4 w-4" />
                        Turn On Camera
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {translation && (
                <div className="flex flex-col space-y-1.5">
                  <h3 className="text-sm font-medium">Translation</h3>
                  <Textarea
                    value={translation}
                    readOnly
                    className="min-h-[100px]"
                  />
                </div>
              )}

              {feedback && (
                <div className="flex flex-col space-y-1.5">
                  <h3 className="text-sm font-medium">Feedback</h3>
                  <Textarea
                    value={feedback}
                    readOnly
                    className="min-h-[100px]"
                  />
                </div>
              )}

              {score !== null && (
                <div className="flex flex-col space-y-1.5">
                  <h3 className="text-sm font-medium">Score</h3>
                  <div className="p-2 bg-gray-100 rounded-md">
                    {(score * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={startTranslation}
              disabled={isTranslating || !isCameraOn}
              className="w-full"
            >
              {isTranslating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                'Start Translation'
              )}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </main>
  );
} 