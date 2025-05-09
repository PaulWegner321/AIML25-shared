#!/bin/bash

# Start the Python backend server
echo "Starting Python backend server..."
python3 backend/llm_service.py &

# Wait a moment for the backend to start
sleep 2

# Start the Next.js frontend
echo "Starting Next.js frontend..."
npm run dev 