interface DescriptionBoxProps {
  description: {
    word: string;
    description: string;
    steps: string[];
    tips: string[];
  };
}

const DescriptionBox = ({ description }: DescriptionBoxProps) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4">How to Sign &ldquo;{description.word}&rdquo;</h2>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Description</h3>
        <p className="text-gray-700">{description.description}</p>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Steps</h3>
        <ol className="list-decimal list-inside space-y-2 text-gray-700">
          {description.steps.map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ol>
      </div>

      <div>
        <h3 className="text-lg font-medium mb-2">Tips</h3>
        <ul className="list-disc list-inside space-y-2 text-gray-700">
          {description.tips.map((tip, index) => (
            <li key={index}>{tip}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default DescriptionBox; 