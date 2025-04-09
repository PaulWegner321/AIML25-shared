export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'https://asl-translate-backend.onrender.com',
  ENDPOINTS: {
    TRANSLATE: '/translate',
    JUDGE: '/judge'
  }
}; 