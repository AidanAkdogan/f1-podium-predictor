#!/bin/bash

echo "🏎️  Starting F1 Podium Predictor"
echo "================================"
echo ""

# Check if model exists
if [ ! -f "models/f1_ranker_v1.pkl" ]; then
    echo "❌ Model not found!"
    echo ""
    echo "Please train the model first:"
    echo "  1. Cache data:  python scripts/warm_cache.py --seasons 2022 2023 2024"
    echo "  2. Train model: python scripts/train_all_races.py --seasons 2022 2023 2024"
    exit 1
fi

echo "✅ Model found"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend
echo "🔧 Starting Flask backend on http://localhost:5000..."
python backend/app.py &
BACKEND_PID=$!
sleep 3

# Start frontend
echo "⚛️  Starting React frontend on http://localhost:3000..."
npm start &
FRONTEND_PID=$!

echo ""
echo "================================"
echo "✅ Both servers running!"
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "================================"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
