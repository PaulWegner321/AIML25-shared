import { NextRequest, NextResponse } from 'next/server';

// Mock translation function for production
function getMockTranslation(tokens: string[]) {
  // Simple mock that joins tokens with spaces and adds some context
  const translation = tokens.join(' ').toLowerCase();
  return `Translation: ${translation}`;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { tokens } = body;

    if (!tokens || !Array.isArray(tokens)) {
      return NextResponse.json(
        { error: 'Invalid request. Tokens must be an array.' },
        { status: 400 }
      );
    }

    // In production, return mock responses
    if (process.env.NODE_ENV === 'production') {
      return NextResponse.json({
        translation: getMockTranslation(tokens)
      });
    }

    // In development, call the real backend API
    const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${BACKEND_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ tokens }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Backend API error:', errorData);
      return NextResponse.json(
        { error: errorData.detail || 'Failed to translate tokens. Please make sure the backend server is running.' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Translation error:', error);
    if (process.env.NODE_ENV === 'production') {
      // In production, return a mock response even on error
      return NextResponse.json({
        translation: 'Sorry, I could not understand those signs. Please try again.'
      });
    }
    return NextResponse.json(
      { error: 'Internal server error. Please make sure the backend server is running.' },
      { status: 500 }
    );
  }
} 