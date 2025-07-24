#!/usr/bin/env python3
"""
Generate Semantic Chunks Prompt

This script asks Gemini to generate the optimal prompt for extracting semantically 
meaningful chunks from scientific papers for vector database embedding.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.settings import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ask_gemini_for_prompt() -> str:
    """Ask Gemini to generate the optimal prompt for semantic chunking."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = """
        I need to extract semantically meaningful chunks from scientific papers for embedding into a vector database. 

        For each chunk, I need to capture:
        1. The actual text content of the semantic chunk
        2. The source paper title
        3. The publication year
        4. The section header where it appears (Introduction, Discussion, Results, etc.)
        5. A brief description of the semantic topic/idea
        6. Page estimate (from section headers with page numbers like "[PAGE 3] INTRODUCTION")
        7. Character position information (start_char and end_char)

        The output should be a JSON array of objects with this structure:
        [
          {
            "text": "The actual text content of the semantic chunk",
            "source_title": "Full paper title",
            "year": 2023,
            "section_header": "Introduction/Discussion/Results/etc",
            "semantic_topic": "Brief description of the main idea/topic",
            "page_estimate": 5,
            "start_char": 10230,
            "end_char": 10850
          }
        ]

        Please create the optimal prompt that I should use to ask an LLM to extract these semantic chunks from scientific paper text. The prompt should:

        1. Clearly explain what semantic chunking means in this context
        2. Provide specific guidelines for what makes a good chunk
        3. Explain how to handle section headers and page numbers
        4. Give clear instructions about the JSON output format
        5. Include examples of good vs bad chunking
        6. Address common challenges like overlapping content, incomplete ideas, etc.
        7. Ensure the chunks will be useful for vector database search and retrieval

        Return ONLY the prompt text that I should use, no additional commentary or explanations.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            logger.info("✅ Successfully generated semantic chunking prompt")
            return response.text.strip()
        else:
            logger.error("Empty response from Gemini")
            return ""
        
    except Exception as e:
        logger.error(f"Failed to generate prompt: {e}")
        return ""

def main():
    """Main function to generate the semantic chunking prompt."""
    logger.info("🚀 Asking Gemini to generate optimal semantic chunking prompt")
    
    prompt = ask_gemini_for_prompt()
    
    if prompt:
        # Save the generated prompt to a file
        prompt_file = Path(__file__).parent / "generated_semantic_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        logger.info(f"✅ Generated prompt saved to: {prompt_file}")
        logger.info("Generated prompt:")
        print("\n" + "="*80)
        print("GENERATED SEMANTIC CHUNKING PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80)
    else:
        logger.error("❌ Failed to generate prompt")

if __name__ == "__main__":
    main() 