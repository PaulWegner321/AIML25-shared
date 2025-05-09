import { useState } from 'react';
import ChatBox from './ChatBox';

interface DescriptionBoxProps {
  description: {
    word: string;
    description: string;
    steps: string[];
    tips: string[];
  };
}

const DescriptionBox = ({ description }: DescriptionBoxProps) => {
  const [showChat, setShowChat] = useState(false);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-2xl font-bold">How to Sign &ldquo;{description.word}&rdquo;</h2>
        <button
          onClick={() => setShowChat(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          Chat with ASL Tutor
        </button>
      </div>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Description</h3>
        <p className="text-gray-700">{description.description}</p>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Steps</h3>
        <div className="space-y-2 text-gray-700">
          {description.steps.map((step, index) => (
            <p key={index} className="ml-4">{step}</p>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-medium mb-2">Tips</h3>
        <ul className="list-disc list-inside space-y-2 text-gray-700">
          {description.tips.map((tip, index) => (
            <li key={index}>{tip}</li>
          ))}
        </ul>
      </div>

      {/* Chat Pop-up */}
      {showChat && (
        <div className="fixed bottom-4 right-4 z-50">
          <div className="w-96 h-[500px] shadow-2xl rounded-lg overflow-hidden">
            <ChatBox signName={description.word} onClose={() => setShowChat(false)} />
          </div>
        </div>
      )}
    </div>
  );
};

export default DescriptionBox; 