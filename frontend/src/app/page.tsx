import Link from 'next/link';

export default function LandingPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold mb-4 text-blue-600">ASL Learning Platform</h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Learn American Sign Language through interactive practice and detailed sign descriptions.
          Our platform helps you master ASL at your own pace with real-time feedback.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-16">
        <div className="bg-white rounded-lg shadow-lg p-8 border border-gray-100 hover:shadow-xl transition-shadow">
          <h2 className="text-2xl font-bold mb-4 text-blue-600">Practice Mode</h2>
          <p className="text-gray-600 mb-6">
            Practice ASL signs with real-time feedback. Our camera-based evaluation helps you perfect your signing technique.
          </p>
          <ul className="space-y-2 mb-6 text-gray-700">
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              Real-time sign evaluation
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              Immediate feedback on your technique
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              Practice at your own pace
            </li>
          </ul>
          <Link 
            href="/practice" 
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Start Practicing
          </Link>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-8 border border-gray-100 hover:shadow-xl transition-shadow">
          <h2 className="text-2xl font-bold mb-4 text-blue-600">Lookup Mode</h2>
          <p className="text-gray-600 mb-6">
            Search for detailed descriptions of ASL signs. Learn how to perform each sign with step-by-step instructions.
          </p>
          <ul className="space-y-2 mb-6 text-gray-700">
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              Comprehensive sign descriptions
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              Step-by-step instructions
            </li>
            <li className="flex items-start">
              <span className="text-green-500 mr-2">✓</span>
              Helpful tips for correct signing
            </li>
          </ul>
          <Link 
            href="/lookup" 
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Look Up Signs
          </Link>
        </div>
      </div>

      <div className="bg-blue-50 rounded-lg p-8 border border-blue-100">
        <h2 className="text-2xl font-bold mb-4 text-blue-700">About This Platform</h2>
        <p className="text-gray-700 mb-4">
          Our ASL Learning Platform is designed to make learning American Sign Language accessible and interactive.
          Whether you&apos;re a beginner or looking to improve your skills, our tools provide the guidance you need.
        </p>
        <p className="text-gray-700">
          The platform uses advanced computer vision to evaluate your signing technique and provides detailed
          descriptions to help you learn new signs. Start your ASL learning journey today!
        </p>
      </div>
    </div>
  );
}
