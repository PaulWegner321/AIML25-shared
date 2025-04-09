# ASL Translation Platform

A full-stack application for American Sign Language (ASL) detection, translation, and evaluation. The platform uses computer vision to detect ASL signs from video input and provides real-time translation and feedback.

## Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── app/                # Application code
│   │   ├── main.py        # FastAPI application and endpoints
│   │   └── models/        # ML models (to be implemented)
│   ├── tests/             # Backend tests
│   ├── requirements.txt   # Python dependencies
│   └── .env.production    # Production environment variables
│
└── nextjs-app/            # Next.js frontend
    ├── src/
    │   ├── app/          # Next.js pages
    │   │   └── video/    # Video translation page
    │   ├── components/   # Reusable UI components
    │   └── config/       # Configuration files
    ├── public/           # Static assets
    └── package.json      # Node.js dependencies
```

## Features

- Real-time video capture and processing
- ASL sign detection (to be implemented)
- Translation of ASL to text
- Performance evaluation and feedback
- Modern, responsive UI
- Secure API communication

## Tech Stack

### Frontend
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- Shadcn UI Components

### Backend
- FastAPI
- Python 3.9+
- IBM Watson Machine Learning
- LangChain
- FAISS
- ChromaDB

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- Git

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd nextjs-app
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```
   ENVIRONMENT=development
   FRONTEND_URL=http://localhost:3000
   PORT=8000
   ```

5. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Deployment

### Frontend (Vercel)
1. Push changes to GitHub
2. Connect repository to Vercel
3. Configure environment variables
4. Deploy

### Backend (Render)
1. Push changes to GitHub
2. Create new Web Service on Render
3. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   ```
   ENVIRONMENT=production
   FRONTEND_URL=<your-frontend-url>
   PORT=10000
   ```

## API Endpoints

### Translation
- **POST** `/translate`
  - Request: `{ "tokens": ["sign1", "sign2", ...] }`
  - Response: `{ "translation": "translated text" }`

### Evaluation
- **POST** `/judge`
  - Request: `{ "translation": "text", "tokens": ["sign1", "sign2", ...] }`
  - Response: `{ "feedback": "feedback text", "score": 0.75 }`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IBM Watson for ML capabilities
- LangChain for RAG pipeline
- FAISS and ChromaDB for vector storage
- The ASL community for sign language resources 