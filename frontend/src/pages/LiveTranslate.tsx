import { useState, useRef, useEffect } from 'react';

export function LiveTranslate() {
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedText, setTranslatedText] = useState('');
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
    // Mock translation for now
    setTranslatedText('Hello, how are you today?');
  };

  const stopTranslation = () => {
    setIsTranslating(false);
    setTranslatedText('');
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
        </div>
      </div>
    </div>
  );
} 