#!/usr/bin/env python3
"""
CI/CD-friendly script to test OpenAI API key availability.
"""

import os
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import OPENAI_API_KEY
from openai import OpenAI

def test_api_key():
    """Test if OpenAI API key is available and valid."""
    
    print("🔑 Testing OpenAI API Key")
    print("=" * 40)
    
    # Check if API key is set
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
        print("❌ OpenAI API key not found or not configured")
        print()
        print("📋 To fix this:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Edit the .env file in the project root")
        print("3. Replace 'your-api-key-here' with your actual API key")
        print("4. Your API key should start with 'sk-'")
        print()
        print("💡 Example .env file:")
        print("OPENAI_API_KEY=sk-1234567890abcdef...")
        print()
        return False
    
    # Check if API key format looks correct
    if not OPENAI_API_KEY.startswith('sk-'):
        print("❌ API key format looks incorrect")
        print("   API key should start with 'sk-'")
        print(f"   Found: {OPENAI_API_KEY[:10]}...")
        return False
    
    print(f"✅ API key found: {OPENAI_API_KEY[:10]}...")
    
    # Test the API key with a simple request
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Make a minimal test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello'"}
            ],
            max_tokens=5
        )
        
        print("✅ API key is valid and working!")
        print(f"   Test response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "authentication" in error_msg or "invalid" in error_msg:
            print("❌ API key is invalid or expired")
            print("   Please check your API key at: https://platform.openai.com/api-keys")
        elif "rate limit" in error_msg:
            print("❌ Rate limit exceeded")
            print("   You may need to upgrade your OpenAI plan")
        else:
            print(f"❌ API test failed: {e}")
            print("   Please check your internet connection and try again")
        return False

def main():
    """Main function to test API key."""
    success = test_api_key()
    
    if success:
        print("\n🎉 API key is ready for use!")
        print("   You can now run: python scripts/test_llm_extraction.py")
    else:
        print("\n🔧 Please fix the API key issue before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main() 