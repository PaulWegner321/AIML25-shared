import { NextResponse } from 'next/server';

// ASL Sign Knowledge Base
const Sign_knowledge = {
  "A": "Thumb: Curled alongside the side of the index finger, resting against it. Index: Bent downward into the palm, creating a firm curve. Middle: Bent downward in line with the index. Ring: Bent downward. Pinky: Bent downward. Palm Orientation: Facing forward (away from your body). Wrist/Forearm: Neutral position; elbow bent naturally. Movement: None. Note: Represents the shape of a capital 'A'.",
  // ... Add other letters similarly
};

const createPrompt = (signName: string): string => {
  const signDetails = Sign_knowledge[signName.toUpperCase()] || "Sign not found.";

  return (
    `You are an American Sign Language (ASL) teacher.\n\n` +
    `Please clearly explain how to perform the ASL sign on a beginner level for the letter '${signName}'. ` +
    `Use simple language and full sentences. Do not assume any prior knowledge about ASL.\n\n` +
    `Here is relevant information for the letter '${signName}':\n` +
    `${signDetails}\n\n` +
    `If you cant generate a description based on the relevant information, output: 'Sorry, I cant help your with this sign' \n\n`
  );
};

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