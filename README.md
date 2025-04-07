# ASL Translation Platform

A full-stack application for real-time American Sign Language (ASL) translation using IBM Watsonx.ai as the LLM backbone.

## Features

- Real-time ASL video input processing
- Watsonx.ai-powered translation
- Translation quality assessment
- Extensible RAG pipeline for context-aware translations
- Modern, responsive UI
- Vercel deployment for seamless hosting

## Prerequisites

- Node.js 18+ (for local frontend development)
- Python 3.9+ (for local backend development)
- IBM Cloud account with Watsonx.ai access
- Vercel account (for deployment)

## Project Structure

```
.
├── frontend/           # React + TypeScript frontend
├── backend/           # FastAPI backend
├── models/            # ASL processing and translation modules
├── config/            # Configuration files
└── vector_db/         # Vector database for RAG
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd asl-translator
   ```

2. Create a `.env` file in the root directory:
   ```
   WATSONX_API_KEY=your_api_key
   WATSONX_PROJECT_ID=your_project_id
   WATSONX_MODEL_ID=your_model_id
   ```

## Local Development

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## Deployment to Vercel

1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)

2. Go to the [Vercel Dashboard](https://vercel.com/dashboard) and import your repository

3. Configure the project settings:
   - Framework Preset: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Install Command: `npm install`

4. Add your environment variables in the Vercel project settings:
   - WATSONX_API_KEY
   - WATSONX_PROJECT_ID
   - WATSONX_MODEL_ID

5. Deploy your project

## API Endpoints

### Translation
- `POST /translate`
  - Request: `{ "tokens": ["HELLO", "THANK", "YOU"] }`
  - Response: `{ "translated_text": "Hello, thank you" }`

### Translation Quality
- `POST /judge`
  - Request: `{ "tokens": ["HELLO"], "translated_text": "Hello" }`
  - Response: `{ "score": 8.5, "suggestions": "Good translation" }`

## Adding Custom ASL Models

1. Implement your model in `models/asl_detector.py`
2. Update the translation pipeline in `models/translator.py`
3. Add model configuration in `config/watsonx_client.py`

## RAG Integration

The system includes a RAG pipeline for context-aware translations:

1. Add your documents to the vector database:
   ```python
   # Example code for adding documents
   ```

2. Configure the RAG settings in `config/rag_config.py`

3. The system will automatically use the RAG pipeline for translations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IBM Watsonx.ai for the LLM backbone
- MediaPipe for hand tracking capabilities
- The open-source community for various tools and libraries