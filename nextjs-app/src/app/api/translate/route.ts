import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { tokens } = body;

    // In a real implementation, this would call your backend API
    // For now, we'll just return a mock response
    const translatedText = tokens.join(' ').toLowerCase();

    return NextResponse.json({ translated_text: translatedText });
  } catch (error) {
    console.error('Error processing translation request:', error);
    return NextResponse.json(
      { error: 'Failed to process translation request' },
      { status: 500 }
    );
  }
} 