// Get the environment
const isDevelopment = process.env.NODE_ENV === 'development';

// Use local backend in development, production backend otherwise
const API_URL = isDevelopment 
  ? 'http://localhost:8000'
  : 'https://asl-translate-backend.onrender.com';

console.log('Using API URL:', API_URL);

export const API_ENDPOINTS = {
  evaluateSign: `${API_URL}/evaluate-sign`,
  signDescription: `${API_URL}/sign-description`,
}; 