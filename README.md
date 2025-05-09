# ASL Learning Platform

An interactive web platform for learning American Sign Language (ASL) through practice and lookup features.

## Project Overview  
This web application is an AI-powered learning tool that helps users master the American Sign Language (ASL) alphabet. A custom Convolutional Neural Network (CNN) provides real-time recognition of hand-signed letters via webcam, while two large vision-language models supply explanatory feedback. Learners see immediate correctness checks, short coaching hints, and can query a built-in RAG system for authoritative written descriptions of any sign. The frontend runs on Next.js/Vercel; the FastAPI backend, deployed on Render, hosts the models and APIs.

## Features

- **Flashcard Practice**: Practice ASL signs with real-time feedback
- **Sign Lookup**: Search for detailed descriptions of ASL signs
- **Interactive Camera**: Real-time sign evaluation using your webcam
- **Detailed Instructions**: Step-by-step guides for performing signs
- **Tips & Feedback**: Get helpful tips and immediate feedback on your signing
- **Computer Vision Detection**: CNN-based ASL sign detection with MediaPipe hand tracking
- **AI-Generated Feedback**: Leverages GPT-4V and Mistral for detailed sign analysis

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
- OpenCV for image processing
- MediaPipe for hand tracking
- PyTorch for CNN model inference
- GPT-4 Vision for image analysis
- Mistral for feedback generation

## Project Structure

```
.
├── frontend/                     # Next.js frontend application
│   ├── src/
│   │   ├── app/                  # Next.js app router pages
│   │   ├── components/           # React components
│   │   └── styles/               # Global styles
│   └── public/                   # Static assets
│
└── backend/                     # FastAPI backend application
    ├── app/
    │   ├── models/               # ML models
    │   │   ├── weights/          # Model weights storage
    │   │   │   ├── cnn_model.pth  # CNN model weights
    │   │   │   └── new_cnn_model.pth  # New CNN model weights
    │   │   ├── new_cnn_model.py  # CNN model architecture
    │   │   └── keypoint_detector.py  # Hand detection models
    │   ├── services/             # AI services and business logic
    │   ├── schemas/              # Pydantic schemas
    │   ├── config/               # Configuration files
    │   └── main.py               # Main FastAPI application
    ├── render.yaml               # Render deployment configuration
    └── requirements.txt          # Python dependencies
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

4. Make sure you have the model weights in the correct location:
   ```bash
   mkdir -p app/models/weights
   # Place your cnn_model.pth and new_cnn_model.pth in this directory
   ```

5. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

6. The API will be available at [http://localhost:8000](http://localhost:8000).

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
4. The render.yaml file includes the necessary configuration for deployment:
   - Automatically installs dependencies
   - Creates necessary directories
   - Copies model weights to the appropriate location
   - Sets environment variables
   - Provides health checks

## AI Models

### CNN Model for ASL Detection

The application uses a Convolutional Neural Network (CNN) for ASL sign detection:

- **Architecture**: Custom CNN with 4 convolutional layers
- **Input**: 224x224 RGB images of hand signs
- **Pre-processing**: MediaPipe hand tracking for better localization
- **Output**: 25 ASL letters (excluding J and Z which require motion)

### Vision and LLM Integration

- **GPT-4 Vision**: Analyzes hand images for detailed description
- **Mistral**: Generates structured feedback based on sign analysis
- **Pipeline**: Images are processed through CNN → GPT-4V → Mistral for comprehensive feedback

## API Endpoints

### Sign Evaluation
- `POST /evaluate`: Evaluates ASL signs using the CNN model + LLM pipeline
- `POST /evaluate-vision`: Vision-based sign analysis
- `POST /evaluate-gpt4o`: GPT-4 Vision analysis of signs
- `POST /predict`: Direct CNN model prediction

### Sign Description
- `POST /sign-description`: Provides detailed sign description
- `POST /lookup`: Searches for ASL sign information

## Future Enhancements

- Real-time sign detection improvements
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