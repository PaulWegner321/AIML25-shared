import { useState } from 'react';

export function Settings() {
  const [settings, setSettings] = useState({
    apiKey: '',
    projectId: '',
    modelId: '',
    language: 'en',
    confidence: 0.8,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: name === 'confidence' ? parseFloat(value) : value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Save settings to backend/localStorage
    console.log('Settings saved:', settings);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Settings</h1>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">API Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700">
                Watsonx API Key
              </label>
              <input
                type="password"
                id="apiKey"
                name="apiKey"
                value={settings.apiKey}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label htmlFor="projectId" className="block text-sm font-medium text-gray-700">
                Project ID
              </label>
              <input
                type="text"
                id="projectId"
                name="projectId"
                value={settings.projectId}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label htmlFor="modelId" className="block text-sm font-medium text-gray-700">
                Model ID
              </label>
              <input
                type="text"
                id="modelId"
                name="modelId"
                value={settings.modelId}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Translation Settings</h2>
          
          <div className="space-y-4">
            <div>
              <label htmlFor="language" className="block text-sm font-medium text-gray-700">
                Target Language
              </label>
              <select
                id="language"
                name="language"
                value={settings.language}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
              </select>
            </div>
            
            <div>
              <label htmlFor="confidence" className="block text-sm font-medium text-gray-700">
                Confidence Threshold
              </label>
              <input
                type="range"
                id="confidence"
                name="confidence"
                min="0"
                max="1"
                step="0.1"
                value={settings.confidence}
                onChange={handleChange}
                className="mt-1 block w-full"
              />
              <div className="text-sm text-gray-500 mt-1">
                Current value: {settings.confidence}
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Save Settings
          </button>
        </div>
      </form>
    </div>
  );
} 