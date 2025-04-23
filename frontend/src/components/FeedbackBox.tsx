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
          <div 
            className={`p-4 rounded-lg ${
              isCorrect ? 'bg-green-100 text-green-900' : 'bg-red-100 text-red-900'
            }`}
            title="The letter that was detected from your sign"
          >
            <h3 className="text-sm font-medium mb-1">Detected Letter</h3>
            <p className="text-2xl font-bold">{detectedLetter}</p>
          </div>

          {/* Confidence Score Box */}
          <div 
            className={`p-4 rounded-lg ${
              isCorrect ? 'bg-green-100 text-green-900' : 'bg-red-100 text-red-900'
            }`}
            title="How confident the model is in its detection (0-100%)"
          >
            <h3 className="text-sm font-medium mb-1">Confidence</h3>
            <p className="text-2xl font-bold">
              {`${(confidence * 100).toFixed(1)}%`}
            </p>
          </div>
        </div>
      )}

      {/* LLM Feedback Box - Only shown when incorrect */}
      {feedback && !isCorrect && (
        <div 
          className="bg-red-100 text-red-900 p-4 rounded-lg mb-4"
          title="Detailed feedback about your sign and suggestions for improvement"
        >
          <h3 className="text-sm font-medium mb-2">Improvement Feedback</h3>
          <p className="font-medium">{feedback}</p>
        </div>
      )}

      {/* General Tips Box - Always shown */}
      <div 
        className="bg-blue-50 p-4 rounded-lg"
        title="General tips for better sign detection"
      >
        <h3 className="font-medium text-blue-800 mb-2">General Tips:</h3>
        <ul className="list-disc list-inside text-blue-700 space-y-1">
          <li>Make sure your hand is clearly visible</li>
          <li>Keep your fingers straight and together</li>
          <li>Position your hand at chest level</li>
          <li>Ensure good lighting</li>
        </ul>
      </div>
    </div>
  );
};

export default FeedbackBox; 