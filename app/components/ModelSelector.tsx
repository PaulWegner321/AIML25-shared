'use client';

import React from 'react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelSelect: (model: string) => void;
}

export default function ModelSelector({ selectedModel, onModelSelect }: ModelSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <label htmlFor="model-select" className="font-medium">
        Model:
      </label>
      <select
        id="model-select"
        value={selectedModel}
        onChange={(e) => onModelSelect(e.target.value)}
        className="p-2 border rounded"
      >
        <option value="cnn">CNN Model</option>
        <option value="gpt4v">GPT-4V</option>
      </select>
    </div>
  );
} 