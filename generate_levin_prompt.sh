#!/bin/bash

# Generate Michael Levin Personality Prompt
# This script uses Gemini to generate the optimal prompt for impersonating Michael Levin

echo " Generating Michael Levin Personality Prompt..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please create one with your API keys:"
    echo "GOOGLE_API_KEY=your_google_api_key_here"
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    exit 1
fi

# Load environment variables
source .env

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY not found in .env file"
    echo "Please add it to your .env file:"
    echo "GOOGLE_API_KEY=your_google_api_key_here"
    exit 1
fi

echo "✅ Environment check passed"
echo "🔧 Running prompt generation script..."

# Run the Python script
uv run python scripts/generate_levin_prompt.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully generated Michael Levin personality prompt!"
    echo "📁 Check scripts/generated_levin_prompt.txt for the generated prompt"
    echo ""
    echo "Next steps:"
    echo "1. Review the generated prompt"
    echo "2. Update the get_conversational_response function in app.py"
    echo "3. Test the improved conversational interface"
else
    echo "❌ Failed to generate prompt"
    exit 1
fi 