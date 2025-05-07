// Get the environment
const isDevelopment = process.env.NODE_ENV === 'development';

// Use local backend in development, production backend otherwise
const API_URL = isDevelopment 
  ? 'http://localhost:8000'
  : 'https://asl-translate-backend.onrender.com';

console.log('Using API URL:', API_URL);

export const API_ENDPOINTS = {
  evaluateSign: `${API_URL}/evaluate-sign`,
  evaluateVision: `${API_URL}/evaluate-vision`,
  signDescription: `${API_URL}/sign-description`,
  debugImages: `${API_URL}/debug-images`,
  debugImage: (filename: string) => `${API_URL}/debug-image/${filename}`,
  diagnostic: `${API_URL}/diagnostic`,
  ping: `${API_URL}/ping`,
}; 