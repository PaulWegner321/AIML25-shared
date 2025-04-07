import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { tokens, translated_text } = body;

    // In a real implementation, this would call your backend API
    // For now, we'll just return a mock response
    const score = Math.random() * 5 + 5; // Random score between 5 and 10
    const suggestions = 'Good translation, but could be improved by being more precise.';

    return NextResponse.json({ 
      score: parseFloat(score.toFixed(1)), 
      suggestions 
    });
  } catch (error) {
    console.error('Error processing translation quality assessment:', error);
    return NextResponse.json(
      { error: 'Failed to process translation quality assessment' },
      { status: 500 }
    );
  }
} 