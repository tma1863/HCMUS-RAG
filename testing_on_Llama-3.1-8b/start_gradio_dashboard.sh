#!/bin/bash

# HippoRAG Gradio Dashboard Startup Script
# ========================================

echo "🦛 Starting HippoRAG Gradio Dashboard..."
echo "================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

echo "✅ Docker and docker-compose are available"

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Show service status
echo "📊 Service Status:"
docker-compose ps

# Check if Gradio is accessible
echo "🌐 Checking Gradio dashboard..."
if curl -f http://localhost:7860 > /dev/null 2>&1; then
    echo "✅ Gradio dashboard is running!"
    echo ""
    echo "🎉 SUCCESS! HippoRAG Gradio Dashboard is ready!"
    echo "================================================="
    echo "📱 Web Dashboard: http://localhost:7860"
    echo "🔧 Ollama API: http://localhost:11434"
    echo ""
    echo "🚀 Usage Instructions:"
    echo "1. Open http://localhost:7860 in your browser"
    echo "2. Go to 'Session Setup' tab"
    echo "3. Select a dataset (AM, DS, or MCS)"
    echo "4. Click 'Start Session'"
    echo "5. Go to 'Interactive Q&A' tab and start asking questions!"
    echo ""
    echo "🛑 To stop: docker-compose down"
    echo "📋 To view logs: docker-compose logs -f"
else
    echo "⚠️  Gradio dashboard is not accessible yet."
    echo "📋 Check logs with: docker-compose logs -f"
fi

echo "=================================================" 