#!/usr/bin/env python3
"""
Generate Michael Levin Personality Prompt

This script asks Gemini to generate the optimal prompt for impersonating Michael Levin
in a conversational RAG application.
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

def ask_gemini_for_levin_prompt() -> str:
    """Ask Gemini to generate the optimal prompt for impersonating Michael Levin."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = """
        I'm building a conversational RAG (Retrieval-Augmented Generation) application based on the persona of Michael Levin, a renowned developmental biologist and researcher.

        I want the AI responses to sound like they are coming from Michael Levin himself. I will provide relevant semantic chunks from Levin's papers as context.

        Please think about and provide the optimal system prompt/instructions that would best impersonate Michael's knowledge, mannerisms, communication style, and personality traits.

        Consider these aspects of Michael Levin's communication style and personality:

        1. **Interdisciplinary Thinking**: He connects developmental biology, computer science, cognitive science, and AI
        2. **Scale-Free Cognition**: He emphasizes how cognitive processes work across different scales (cells, tissues, organisms)
        3. **Bioelectricity Focus**: He's deeply interested in how electrical networks in cells create cognition and behavior
        4. **Provocative Questioning**: He often asks "what if" questions that challenge conventional thinking
        5. **Enthusiasm for Unconventional Substrates**: He's interested in cognition in plants, cells, and non-neural systems
        6. **Technical Precision**: He uses specific terms like "bioelectricity," "morphogenesis," "basal cognition," "collective intelligence"
        7. **Personal Voice**: He often uses "I" when presenting his own hypotheses and research
        8. **Counterintuitive Ideas**: He presents ideas that challenge conventional wisdom about intelligence and cognition
        9. **Future-Oriented**: He often discusses implications for AI, synthetic biology, and exobiology
        10. **Collaborative Spirit**: He frequently references his collaborators and interdisciplinary work

        The prompt should:
        - Capture his distinctive communication style
        - Include his key concepts and terminology
        - Reflect his enthusiasm and curiosity
        - Maintain his technical precision while being accessible
        - Encourage responses that sound authentically like him
        - Handle citations appropriately (he often references his own work)
        - Balance technical detail with broader implications

        Return ONLY the system prompt/instructions that I should use to make an AI sound like Michael Levin, no additional commentary or explanations.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            logger.info("✅ Successfully generated Michael Levin personality prompt")
            return response.text.strip()
        else:
            logger.error("Empty response from Gemini")
            return ""
        
    except Exception as e:
        logger.error(f"Failed to generate prompt: {e}")
        return ""

def main():
    """Main function to generate the Michael Levin personality prompt."""
    logger.info("🚀 Asking Gemini to generate optimal Michael Levin personality prompt")
    
    prompt = ask_gemini_for_levin_prompt()
    
    if prompt:
        # Save the generated prompt to a file
        prompt_file = Path(__file__).parent / "generated_levin_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        logger.info(f"✅ Generated prompt saved to: {prompt_file}")
        logger.info("Generated prompt:")
        print("\n" + "="*80)
        print("GENERATED MICHAEL LEVIN PERSONALITY PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80)
    else:
        logger.error("❌ Failed to generate prompt")

if __name__ == "__main__":
    main() 