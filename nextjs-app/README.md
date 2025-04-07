# ASL Translator

A real-time American Sign Language (ASL) translation application built with Next.js.

## Features

- Live webcam feed for ASL input
- Modern, responsive UI with Tailwind CSS
- Client-side camera access and video processing
- Placeholder for real-time ASL translation

## Prerequisites

- Node.js 18+ and npm
- Modern web browser with webcam support
- (Optional) Vercel account for deployment

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Development

The application is built with:
- Next.js 13+ (App Router)
- TypeScript
- Tailwind CSS
- Lucide React for icons

### Project Structure

```
src/
├── app/
│   ├── layout.tsx    # Root layout with Sidebar and Header
│   ├── page.tsx      # Live Translation page
│   └── settings/     # Settings page
├── components/       # Reusable components
└── styles/          # Global styles
```

## Deployment

The application is configured for deployment on Vercel:

1. Push your code to GitHub
2. Import the repository in Vercel
3. Deploy with default settings

## License

MIT 