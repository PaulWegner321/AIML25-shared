#!/bin/bash
set -e

echo "Installing dependencies..."
npm install

echo "Building the application..."
npm run build

echo "Listing contents of dist directory:"
ls -la dist/

echo "Build completed successfully!" 