#!/bin/bash

# Probability Distribution Visualizer - Run Script

set -e

echo "=================================================="
echo "   Probability Distribution Visualizer Setup"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "=================================================="
echo "   Ready! Choose an option:"
echo "=================================================="
echo ""
echo "1. Launch Web Interface (Streamlit)"
echo "2. Run Example Scripts"
echo "3. Run Tests"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Streamlit web interface..."
        echo "   The app will open in your browser at http://localhost:8501"
        echo ""
        streamlit run web/app.py
        ;;
    2)
        echo ""
        echo "📊 Running example scripts..."
        python examples/basic_usage.py
        echo ""
        echo "✓ Examples completed! Check the generated PNG files."
        ;;
    3)
        echo ""
        echo "🧪 Running tests..."
        pytest tests/ -v
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

deactivate
