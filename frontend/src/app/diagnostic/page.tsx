'use client';

import { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '@/utils/api';
import Link from 'next/link';

interface DebugImage {
  filename: string;
  timestamp: number;
  url: string;
}

interface DiagnosticData {
  status: string;
  timestamp?: string;
  images?: {
    raw: string;
    grayscale: string | null;
    api_input: string | null;
  };
  api_response?: any;
  message?: string;
}

const DiagnosticPage = () => {
  const [images, setImages] = useState<DebugImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [diagnostic, setDiagnostic] = useState<DiagnosticData | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  
  useEffect(() => {
    // Fetch all debug images
    const fetchImages = async () => {
      try {
        setLoading(true);
        const response = await fetch(API_ENDPOINTS.debugImages);
        const data = await response.json();
        
        if (data.images) {
          setImages(data.images);
        } else {
          setError('No images found');
        }
      } catch (err) {
        setError('Failed to fetch debug images');
        console.error('Error fetching debug images:', err);
      } finally {
        setLoading(false);
      }
    };
    
    // Fetch diagnostic data
    const fetchDiagnostic = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.diagnostic);
        const data = await response.json();
        setDiagnostic(data);
      } catch (err) {
        console.error('Error fetching diagnostic data:', err);
      }
    };
    
    fetchImages();
    fetchDiagnostic();
  }, []);
  
  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };
  
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">ASL Vision Diagnostic</h1>
      
      <div className="mb-8 p-4 bg-blue-50 rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Latest Diagnostic Result</h2>
        {diagnostic ? (
          <div>
            <p><span className="font-semibold">Status:</span> {diagnostic.status}</p>
            {diagnostic.message && (
              <p className="text-red-500">{diagnostic.message}</p>
            )}
            
            {diagnostic.images && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <h3 className="font-medium mb-2">Raw Image (Original)</h3>
                  <img 
                    src={API_ENDPOINTS.debugImage(diagnostic.images.raw.split('/').pop() || '')} 
                    alt="Raw input" 
                    className="w-full h-auto border border-gray-300 rounded-lg"
                  />
                </div>
                
                {diagnostic.images.grayscale && (
                  <div>
                    <h3 className="font-medium mb-2">Grayscale Image</h3>
                    <img 
                      src={API_ENDPOINTS.debugImage(diagnostic.images.grayscale.split('/').pop() || '')} 
                      alt="Grayscale" 
                      className="w-full h-auto border border-gray-300 rounded-lg"
                    />
                  </div>
                )}
                
                {diagnostic.images.api_input && (
                  <div>
                    <h3 className="font-medium mb-2">API Input Image</h3>
                    <img 
                      src={API_ENDPOINTS.debugImage(diagnostic.images.api_input.split('/').pop() || '')} 
                      alt="API input" 
                      className="w-full h-auto border border-gray-300 rounded-lg"
                    />
                  </div>
                )}
              </div>
            )}
            
            {diagnostic.api_response && (
              <div className="mt-4">
                <h3 className="font-medium mb-2">API Response</h3>
                <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                  {JSON.stringify(diagnostic.api_response, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ) : (
          <p>Loading diagnostic data...</p>
        )}
      </div>
      
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-2">Recent Debug Images</h2>
        
        {loading ? (
          <p>Loading images...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {images.map((image) => (
              <div 
                key={image.filename} 
                className="border border-gray-300 rounded-lg overflow-hidden cursor-pointer hover:border-blue-500"
                onClick={() => setSelectedImage(image.filename)}
              >
                <img 
                  src={API_ENDPOINTS.debugImage(image.filename)} 
                  alt={image.filename} 
                  className="w-full h-48 object-cover"
                />
                <div className="p-2 text-xs">
                  <p className="truncate">{image.filename}</p>
                  <p className="text-gray-500">{formatTimestamp(image.timestamp)}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-3xl w-full p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">{selectedImage}</h3>
              <button 
                onClick={() => setSelectedImage(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                Close
              </button>
            </div>
            <img 
              src={API_ENDPOINTS.debugImage(selectedImage)} 
              alt={selectedImage} 
              className="w-full h-auto max-h-[70vh] object-contain"
            />
          </div>
        </div>
      )}
      
      <div className="mt-8">
        <Link href="/practice" className="text-blue-500 hover:underline">
          &larr; Back to Practice
        </Link>
      </div>
    </div>
  );
};

export default DiagnosticPage; 