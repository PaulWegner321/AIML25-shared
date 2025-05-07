# ASL Learning Platform

An interactive web platform for learning American Sign Language (ASL) through practice and lookup features.

## Features

- **Flashcard Practice**: Practice ASL signs with real-time feedback
- **Sign Lookup**: Search for detailed descriptions of ASL signs
- **Interactive Camera**: Real-time sign evaluation using your webcam
- **Detailed Instructions**: Step-by-step guides for performing signs
- **Tips & Feedback**: Get helpful tips and immediate feedback on your signing

## Tech Stack

### Frontend
- Next.js 14 (App Router)
- TypeScript
- TailwindCSS
- React Webcam

### Backend
- FastAPI
- Python 3.11+
- Pydantic
- OpenCV (for future sign detection)

## Project Structure

```
.
├── frontend/                 # Next.js frontend application
│   ├── src/
│   │   ├── app/             # Next.js app router pages
│   │   ├── components/      # React components
│   │   └── styles/          # Global styles
│   └── public/              # Static assets
│
└── backend/                 # FastAPI backend application
    ├── app/
    │   ├── models/          # ML models and business logic
    │   ├── schemas/         # Pydantic schemas
    │   └── config/          # Configuration files
    └── requirements.txt     # Python dependencies
```

## Local Development

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

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

4. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. The API will be available at [http://localhost:8000](http://localhost:8000).

## Deployment

### Frontend (Vercel)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Configure environment variables:
   - `NEXT_PUBLIC_API_URL`: Backend API URL

### Backend (Render)

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your repository
4. Configure environment variables:
   - `ENVIRONMENT`: Production
   - Add any API keys or secrets

## API Endpoints

### Sign Evaluation
- `POST /evaluate-sign`
  - Input: Image file + expected sign
  - Output: Evaluation results with feedback

### Sign Description
- `POST /sign-description`
  - Input: Word to look up
  - Output: Detailed sign description with steps and tips

## Future Enhancements

- Real-time sign detection using OpenCV
- Integration with IBM WatsonX for improved RAG
- User accounts and progress tracking
- Practice session history
- Mobile app version
- Additional ASL learning resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 