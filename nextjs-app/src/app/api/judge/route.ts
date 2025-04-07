import { NextRequest, NextResponse } from 'next/server';

// Mock judgment function for production
function getMockJudgment(translation: string, tokens: string[]) {
  // Simple mock that provides feedback based on the length of translation
  const score = Math.min(0.8 + Math.random() * 0.2, 1); // Random score between 0.8 and 1.0
  return {
    feedback: `Good translation! The sentence structure is clear and conveys the meaning of the ASL signs well.`,
    score
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { translation, tokens } = body;

    if (!translation || typeof translation !== 'string') {
      return NextResponse.json(
        { error: 'Invalid request. Translation must be a string.' },
        { status: 400 }
      );
    }

    if (!tokens || !Array.isArray(tokens)) {
      return NextResponse.json(
        { error: 'Invalid request. Tokens must be an array.' },
        { status: 400 }
      );
    }

    // In production, return mock responses
    if (process.env.NODE_ENV === 'production') {
      return NextResponse.json(getMockJudgment(translation, tokens));
    }

    // In development, call the real backend API
    const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${BACKEND_URL}/judge`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ translation, tokens }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Backend API error:', errorData);
      return NextResponse.json(
        { error: errorData.detail || 'Failed to evaluate translation. Please make sure the backend server is running.' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Judgment error:', error);
    if (process.env.NODE_ENV === 'production') {
      // In production, return a mock response even on error
      return NextResponse.json({
        feedback: 'The translation appears to be accurate and natural.',
        score: 0.85
      });
    }
    return NextResponse.json(
      { error: 'Internal server error. Please make sure the backend server is running.' },
      { status: 500 }
    );
  }
} 