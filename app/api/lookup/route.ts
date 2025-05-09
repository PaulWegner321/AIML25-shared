import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { signName } = await req.json();

    const response = await fetch('http://localhost:8000/lookup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sign_name: signName }),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch from Python backend');
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in lookup API:', error);
    return NextResponse.json(
      { error: 'Failed to get sign description' },
      { status: 500 }
    );
  }
} 