'use client';

import { useState } from 'react';

export default function Settings() {
  const [settings, setSettings] = useState({
    language: 'en',
    model: 'watsonx',
    confidence: 0.7,
    autoStart: false,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setSettings({
      ...settings,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Save settings to localStorage or API
    localStorage.setItem('aslTranslatorSettings', JSON.stringify(settings));
    alert('Settings saved successfully!');
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Settings</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <form onSubmit={handleSubmit}>
          <div className="space-y-6">
            <div>
              <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
                Target Language
              </label>
              <select
                id="language"
                name="language"
                value={settings.language}
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
              </select>
            </div>
            
            <div>
              <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-1">
                Translation Model
              </label>
              <select
                id="model"
                name="model"
                value={settings.model}
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="watsonx">IBM Watsonx.ai</option>
                <option value="gpt">GPT-4</option>
                <option value="custom">Custom Model</option>
              </select>
            </div>
            
            <div>
              <label htmlFor="confidence" className="block text-sm font-medium text-gray-700 mb-1">
                Confidence Threshold: {settings.confidence}
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
                className="w-full"
              />
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoStart"
                name="autoStart"
                checked={settings.autoStart}
                onChange={handleChange}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="autoStart" className="ml-2 block text-sm text-gray-700">
                Auto-start translation when camera is active
              </label>
            </div>
          </div>
          
          <div className="mt-6">
            <button
              type="submit"
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Save Settings
            </button>
          </div>
        </form>
      </div>
    </div>
  );
} 