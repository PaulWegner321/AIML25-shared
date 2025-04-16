interface FeedbackBoxProps {
  feedback: string | null;
  isCorrect: boolean | null;
}

const FeedbackBox = ({ feedback, isCorrect }: FeedbackBoxProps) => {
  if (!feedback) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Feedback</h2>
        <p className="text-gray-500">Sign a letter to get feedback</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4">Feedback</h2>
      
      <div className={`p-4 rounded-lg mb-4 ${
        isCorrect ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
      }`}>
        <p className="font-medium">{feedback}</p>
      </div>

      {!isCorrect && (
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-medium text-blue-800 mb-2">Tips:</h3>
          <ul className="list-disc list-inside text-blue-700 space-y-1">
            <li>Make sure your hand is clearly visible</li>
            <li>Keep your fingers straight and together</li>
            <li>Position your hand at chest level</li>
            <li>Ensure good lighting</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default FeedbackBox; 