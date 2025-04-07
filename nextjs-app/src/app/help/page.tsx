'use client';

export default function Help() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Help & Documentation</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
        <h2 className="text-xl font-semibold mb-4">Getting Started</h2>
        <p className="mb-4">
          The ASL Translator application allows you to translate American Sign Language (ASL) to text in real-time.
          Follow these steps to get started:
        </p>
        <ol className="list-decimal pl-6 space-y-2">
          <li>Click the "Start Camera" button to activate your webcam</li>
          <li>Position your hands in the camera view</li>
          <li>Click "Start Translation" to begin the ASL recognition process</li>
          <li>The translated text will appear in the translation box below</li>
        </ol>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
        <h2 className="text-xl font-semibold mb-4">Tips for Best Results</h2>
        <ul className="list-disc pl-6 space-y-2">
          <li>Ensure good lighting in your environment</li>
          <li>Keep your hands within the camera frame</li>
          <li>Make clear, deliberate hand movements</li>
          <li>Position yourself so your entire upper body is visible</li>
          <li>Avoid wearing gloves or accessories that might interfere with hand tracking</li>
        </ul>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-xl font-semibold mb-4">Frequently Asked Questions</h2>
        
        <div className="space-y-4">
          <div>
            <h3 className="font-medium">What browsers are supported?</h3>
            <p>The application works best in Chrome, Firefox, and Edge. Safari may have limited functionality.</p>
          </div>
          
          <div>
            <h3 className="font-medium">Is my data being stored?</h3>
            <p>No, all processing happens in real-time and no video data is stored. Your settings are saved locally on your device.</p>
          </div>
          
          <div>
            <h3 className="font-medium">Can I use this offline?</h3>
            <p>No, the application requires an internet connection to access the AI models for translation.</p>
          </div>
          
          <div>
            <h3 className="font-medium">How accurate is the translation?</h3>
            <p>The accuracy depends on various factors including lighting, camera quality, and the clarity of your hand movements. For best results, follow the tips above.</p>
          </div>
        </div>
      </div>
    </div>
  );
} 