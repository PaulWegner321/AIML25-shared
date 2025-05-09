// Get the environment
const isDevelopment = process.env.NODE_ENV === 'development';

// Use local backend in development, production backend otherwise
const API_BASE = isDevelopment ? 'http://localhost:8000' : process.env.NEXT_PUBLIC_API_URL || 'https://asl-api.onrender.com';

console.log('Using API URL:', API_BASE);

export const API_ENDPOINTS = {
  evaluateSign: `${API_BASE}/evaluate-vision`,
  evaluateGPT4o: `${API_BASE}/evaluate-gpt4o`,
  evaluate: `${API_BASE}/evaluate`,  // New endpoint for CNN → GPT-4V → Mistral pipeline
  predict: `${API_BASE}/predict`,
  signDescription: `${API_BASE}/sign-description`,
  chat: `${API_BASE}/chat`,  // New endpoint for interactive ASL chat
  debugImages: `${API_BASE}/debug-images`,
  debugImage: (filename: string) => `${API_BASE}/debug-image/${filename}`,
  diagnostic: `${API_BASE}/diagnostic`,
  ping: `${API_BASE}/ping`,
  getFeedback: `${API_BASE}/get-feedback`,
}; 