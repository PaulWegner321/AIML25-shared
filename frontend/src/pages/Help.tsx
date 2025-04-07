import { useState } from 'react';

interface FAQ {
  question: string;
  answer: string;
}

const faqs: FAQ[] = [
  {
    question: 'How does the ASL translation work?',
    answer: 'The system uses computer vision to detect hand gestures and facial expressions, then converts them into ASL tokens. These tokens are processed by the Watsonx AI model to generate natural language translations.'
  },
  {
    question: 'What are the system requirements?',
    answer: 'You need a modern web browser with camera access and a stable internet connection. The system works best with good lighting and a clear view of your hands and face.'
  },
  {
    question: 'How accurate is the translation?',
    answer: 'Translation accuracy depends on various factors including lighting conditions, camera quality, and the complexity of the signs. The system provides confidence scores for each translation.'
  },
  {
    question: 'Can I use the system offline?',
    answer: 'Currently, the system requires an internet connection to access the Watsonx AI model. Offline functionality is planned for future updates.'
  }
];

export function Help() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Help & Documentation</h1>
      
      <div className="space-y-8">
        <section className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Getting Started</h2>
          <div className="prose">
            <p>
              Welcome to the ASL Translation Platform. This tool helps you translate American Sign Language
              (ASL) into text in real-time using your computer's camera.
            </p>
            <h3>Quick Start Guide:</h3>
            <ol>
              <li>Click "Start Camera" to enable your webcam</li>
              <li>Position yourself in front of the camera</li>
              <li>Click "Start Translation" to begin</li>
              <li>Sign naturally, and the translation will appear below</li>
            </ol>
          </div>
        </section>

        <section className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Frequently Asked Questions</h2>
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div key={index} className="border rounded-lg">
                <button
                  className="w-full px-4 py-3 text-left flex justify-between items-center hover:bg-gray-50"
                  onClick={() => setOpenFaq(openFaq === index ? null : index)}
                >
                  <span className="font-medium">{faq.question}</span>
                  <span className="text-gray-500">
                    {openFaq === index ? '−' : '+'}
                  </span>
                </button>
                {openFaq === index && (
                  <div className="px-4 py-3 bg-gray-50 border-t">
                    {faq.answer}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        <section className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Contact Support</h2>
          <p className="mb-4">
            Need additional help? Our support team is here to assist you.
          </p>
          <div className="space-y-2">
            <p>
              <strong>Email:</strong>{' '}
              <a href="mailto:support@asltranslator.com" className="text-blue-500 hover:underline">
                support@asltranslator.com
              </a>
            </p>
            <p>
              <strong>Hours:</strong> Monday - Friday, 9:00 AM - 5:00 PM EST
            </p>
          </div>
        </section>
      </div>
    </div>
  );
} 