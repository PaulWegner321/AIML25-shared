const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  evaluateSign: `${API_URL}/evaluate-sign`,
  signDescription: `${API_URL}/sign-description`,
}; 