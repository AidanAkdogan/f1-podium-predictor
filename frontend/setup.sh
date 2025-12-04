#!/bin/bash

echo "🏎️  F1 Podium Predictor Setup Script"
echo "===================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

echo "✅ Node.js found: $(node --version)"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✅ Python dependencies installed"
echo ""

# Install Node dependencies
echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

echo "✅ Node.js dependencies installed"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p models artifacts data/raw/.fastf1cache

echo "✅ Directories created"
echo ""

echo "===================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Cache F1 data:    python scripts/warm_cache.py --seasons 2022 2023 2024"
echo "2. Train model:      python scripts/train_all_races.py --seasons 2022 2023 2024"
echo "3. Start backend:    python backend/app.py"
echo "4. Start frontend:   npm start"
echo ""
echo "For full instructions, see README.md"
echo "===================================="
