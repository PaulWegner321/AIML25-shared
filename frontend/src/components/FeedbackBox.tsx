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
  expectedLetter?: string | null;
}

const FeedbackBox = ({ feedback, isCorrect, detectedLetter, confidence, expectedLetter }: FeedbackBoxProps) => {
  // Parse the feedback to extract description, steps, and tips
  const parseFeedback = (feedback: string): { description: string, steps: string[], tips: string[] } => {
    const parts = { 
      description: '', 
      steps: [] as string[], 
      tips: [] as string[] 
    };
    
    if (!feedback) return parts;
    
    // Remove placeholder text patterns that shouldn't be displayed
    const cleanFeedback = feedback.replace(/\[Optional:.*?\]/g, '')
                                  .replace(/\[Optional.*?\]/g, '')
                                  .replace(/\[Write.*?\]/g, '')
                                  .replace(/\[First.*?\]/g, '')
                                  .replace(/\[Second.*?\]/g, '')
                                  .replace(/\[Third.*?\]/g, '')
                                  .replace(/\[In the case of.*?\]/g, '')
                                  .replace(/\[.*?provide.*?\]/g, '')
                                  .replace(/\[.*?improvement.*?\]/g, '');
    
    // Split the feedback into sections
    const sections = cleanFeedback.split('\n\n');
    
    // First section is always the description
    if (sections.length > 0) {
      parts.description = sections[0].trim();
      
      // Further clean the description - if it's just a dash or empty, replace it
      if (parts.description === '-' || parts.description === '' || parts.description.length < 3) {
        parts.description = '';
      }
    }
    
    // Find steps section
    const stepsIndex = sections.findIndex(s => s.toLowerCase().includes('steps to improve:'));
    if (stepsIndex !== -1) {
      const stepsText = sections[stepsIndex].replace('Steps to improve:', '').trim();
      // Filter out empty lines and placeholders
      parts.steps = stepsText.split('\n')
                             .filter(s => s.trim() !== '' && !s.includes('[Optional'))
                             .map(s => s.replace(/\[.*?\]/g, '').trim());
    }
    
    // Find tips section
    const tipsIndex = sections.findIndex(s => s.toLowerCase().includes('tips:'));
    if (tipsIndex !== -1) {
      const tipsText = sections[tipsIndex].replace('Tips:', '').trim();
      // Filter out empty lines and placeholders
      parts.tips = tipsText.split('\n')
                           .filter(s => s.trim() !== '' && !s.includes('[Optional'))
                           .map(s => s.replace(/\[.*?\]/g, '').trim());
    }
    
    return parts;
  };
  
  const feedbackParts = feedback ? parseFeedback(feedback) : { description: '', steps: [], tips: [] };
  
  // Debug logging
  console.log("Feedback description:", feedbackParts.description);
  console.log("Is correct:", isCorrect);
  console.log("Description is '-':", feedbackParts.description === '-');
  console.log("Description is empty:", !feedbackParts.description);
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4">Feedback</h2>
      
      {/* Result Boxes - Only shown after evaluation */}
      {detectedLetter !== null && confidence !== null && (
        <div className={`grid ${isCorrect ? 'grid-cols-2' : 'grid-cols-3'} gap-4 mb-6`}>
          {/* Detected Letter Box */}
          <Tooltip text="The letter that was detected from your hand sign">
            <div 
              className={`p-4 rounded-lg relative ${
                isCorrect ? 'bg-green-100 text-green-900' : 'bg-orange-100 text-orange-900'
              }`}
            >
              <h3 className="text-sm font-medium mb-1">Detected Sign</h3>
              <p className="text-2xl font-bold">{detectedLetter}</p>
            </div>
          </Tooltip>

          {/* Expected Letter Box - Only shown when there's a mismatch */}
          {!isCorrect && expectedLetter && (
            <Tooltip text="The letter you were supposed to sign according to the flashcard">
              <div className="p-4 rounded-lg relative bg-blue-100 text-blue-900">
                <h3 className="text-sm font-medium mb-1">Expected Sign</h3>
                <p className="text-2xl font-bold">{expectedLetter}</p>
              </div>
            </Tooltip>
          )}

          {/* Confidence Score Box */}
          <Tooltip text="How confident the model is in its detection (0-100%)">
            <div 
              className={`p-4 rounded-lg relative ${
                isCorrect ? 'bg-green-100 text-green-900' : 'bg-orange-100 text-orange-900'
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

      {/* Structured Feedback Box */}
      {feedback && (
        <Tooltip text={isCorrect ? "Analysis of your correct sign" : "Detailed feedback about your sign and suggestions for improvement"}>
          <div 
            className={`${
              isCorrect ? 'bg-green-50 border border-green-200' : 'bg-orange-50 border border-orange-200'
            } rounded-lg mb-4 relative overflow-hidden`}
          >
            <h3 className={`text-sm font-medium p-3 mb-0 ${
              isCorrect ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'
            }`}>
              {isCorrect ? 'Analysis' : 'Improvement Feedback'}
            </h3>

            {/* Description Section */}
            <div className="p-4 border-b border-gray-200">
              {(!feedbackParts.description || feedbackParts.description === '-' || feedbackParts.description.trim() === '') ? (
                isCorrect ? 
                "Great job! Your hand position is perfect for this sign. You've made excellent progress in mastering this letter." :
                "Your sign needs some adjustment. Try following the tips below to improve your form."
              ) : (
                feedbackParts.description
              )}
            </div>
            
            {/* Steps Section - only show if there are steps */}
            {feedbackParts.steps.length > 0 && (
              <div className="p-4 border-b border-gray-200">
                <h4 className="font-medium mb-2">Steps to improve:</h4>
                <ol className="list-decimal list-inside space-y-1 ml-2">
                  {feedbackParts.steps.map((step, index) => (
                    <li key={index} className="pl-2">{step}</li>
                  ))}
                </ol>
              </div>
            )}
            
            {/* Tips Section - only show if there are tips */}
            {feedbackParts.tips.length > 0 && (
              <div className="p-4">
                <h4 className="font-medium mb-2">Tips:</h4>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  {feedbackParts.tips.map((tip, index) => (
                    <li key={index} className="pl-2">{tip}</li>
                  ))}
                </ul>
              </div>
            )}
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