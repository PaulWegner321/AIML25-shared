interface TooltipProps {
  text: string;
  children: React.ReactNode;
}

const Tooltip = ({ text, children }: TooltipProps) => {
  return (
    <div className="relative w-full">
      <div className="absolute top-2 right-2 z-20">
        <div className="group relative inline-block">
          <div className="w-5 h-5 flex items-center justify-center rounded-full border border-gray-400 cursor-help">
            <span className="text-sm text-gray-500 font-medium">i</span>
          </div>
          <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 absolute z-30 w-64 p-2 text-sm text-white bg-gray-900 rounded-lg -top-2 right-7 pointer-events-none">
            {text}
            <div className="absolute top-3 -left-1 transform -translate-x-1/2 border-4 border-transparent border-r-gray-900"></div>
          </div>
        </div>
      </div>
      {children}
    </div>
  );
};

interface FeedbackBoxProps {
  feedback: string | null;
  isCorrect: boolean | null;
  detectedLetter: string | null;
  confidence: number | null;
}

const FeedbackBox = ({ feedback, isCorrect, detectedLetter, confidence }: FeedbackBoxProps) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4">Feedback</h2>
      
      {/* Result Boxes - Only shown after evaluation */}
      {detectedLetter !== null && confidence !== null && (
        <div className="grid grid-cols-2 gap-4 mb-6">
          {/* Detected Letter Box */}
          <Tooltip text="The letter that was detected from your hand sign">
            <div 
              className={`p-4 rounded-lg relative ${
                isCorrect ? 'bg-green-100 text-green-900' : 'bg-red-100 text-red-900'
              }`}
            >
              <h3 className="text-sm font-medium mb-1">Detected Letter</h3>
              <p className="text-2xl font-bold">{detectedLetter}</p>
            </div>
          </Tooltip>

          {/* Confidence Score Box */}
          <Tooltip text="How confident the model is in its detection (0-100%)">
            <div 
              className={`p-4 rounded-lg relative ${
                isCorrect ? 'bg-green-100 text-green-900' : 'bg-red-100 text-red-900'
              }`}
            >
              <h3 className="text-sm font-medium mb-1">Confidence</h3>
              <p className="text-2xl font-bold">
                {`${(confidence * 100).toFixed(1)}%`}
              </p>
            </div>
          </Tooltip>
        </div>
      )}

      {/* Feedback Box - Shown for both correct and incorrect signs */}
      {feedback && (
        <Tooltip text={isCorrect ? "Analysis of your correct sign" : "Detailed feedback about your sign and suggestions for improvement"}>
          <div 
            className={`${
              isCorrect ? 'bg-green-100 text-green-900' : 'bg-red-100 text-red-900'
            } p-4 rounded-lg mb-4 relative`}
          >
            <h3 className="text-sm font-medium mb-2">
              {isCorrect ? 'Analysis' : 'Improvement Feedback'}
            </h3>
            <p className="font-medium">{feedback}</p>
          </div>
        </Tooltip>
      )}

      {/* General Tips Box - Always shown */}
      <Tooltip text="General tips to help you get better sign detection results">
        <div className="bg-blue-50 p-4 rounded-lg relative">
          <h3 className="font-medium text-blue-800 mb-2">General Tips:</h3>
          <ul className="list-disc list-inside text-blue-700 space-y-1">
            <li>Make sure your hand is clearly visible</li>
            <li>Keep your fingers straight and together</li>
            <li>Position your hand at chest level</li>
            <li>Ensure good lighting</li>
          </ul>
        </div>
      </Tooltip>
    </div>
  );
};

export default FeedbackBox; 