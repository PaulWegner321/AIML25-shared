export async function processFrame(
  videoElement: HTMLVideoElement | null,
  canvasElement: HTMLCanvasElement | null,
  selectedModel: string
): Promise<string> {
  if (!videoElement || !canvasElement) {
    return '';
  }

  const context = canvasElement.getContext('2d');
  if (!context) {
    return '';
  }

  // Draw the current video frame onto the canvas
  context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  // Get the image data from the canvas
  const imageData = canvasElement.toDataURL('image/jpeg');

  try {
    // Send the image data to the appropriate model endpoint
    const endpoint = selectedModel === 'cnn' ? '/api/cnn' : '/api/gpt4v';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) {
      throw new Error('Failed to process frame');
    }

    const data = await response.json();
    return data.feedback || '';
  } catch (error) {
    console.error('Error processing frame:', error);
    return 'Error processing frame. Please try again.';
  }
} 